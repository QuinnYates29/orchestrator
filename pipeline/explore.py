"""Read-only research fan-out: split a question into sub-questions, answer
them concurrently with read-only tools, then synthesize a final answer.

This is the cheapest workflow in the harness — no clones, no writes, no
supervisor, no merge. It fans out perfectly because each sub-question can
be answered independently by reading the repo.

Three stages:

1. **Split.** One call to the explorer role model. It lists the repo root
   and produces 2-6 independent sub-questions (or 1 if the question is
   simple).
2. **Answer.** One agent per sub-question, concurrently, bounded by
   `limits.max_concurrent_workers`. Each gets `READ_ONLY_TOOL_SCHEMAS`
   against the real repo directory. No writes, no run_shell. Turn budget
   is lower than a worker's — 15 by default.
3. **Synthesize.** One call giving the original question and every
   sub-answer, producing the final answer with file-level attribution.

A sub-agent that fails must not sink the run: record the failure, keep
the others, and let the synthesis stage report the gap.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from .client import OrchestratorClient, OrchestratorError
from .config import PipelineCfg
from .events import EventLog, NullEventLog
from .tokens import estimate_usage
from .tools import READ_ONLY_TOOL_SCHEMAS, execute_tool

log = logging.getLogger("pipeline.explore")


# -- data model ---------------------------------------------------------

@dataclass
class SubQuestionResult:
    question: str
    answer: str = ""
    stop_reason: str = ""          # "finished" | "max_turns" | "error"
    tool_calls: int = 0
    turns: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None
    ok: bool = False              # True if a non-empty answer was produced


@dataclass
class ExploreResult:
    question: str
    sub_questions: list[str]
    sub_answers: list[SubQuestionResult]
    final_answer: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_s: float = 0.0
    # Token totals broken down by stage for observability.
    split_prompt_tokens: int = 0
    split_completion_tokens: int = 0
    synthesize_prompt_tokens: int = 0
    synthesize_completion_tokens: int = 0
    messages: list[dict] = field(default_factory=list)


# -- tool schemas -------------------------------------------------------

SUBMIT_QUESTIONS_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_questions",
        "description": "Submit the list of sub-questions derived from the original question. "
                       "Each sub-question should be independently answerable by reading the repo. "
                       "If the question is simple enough that one sub-question fully answers it, "
                       "submit exactly one. Do not pad to hit a target number.",
        "parameters": {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of 1-6 independent sub-questions derived from the original "
                                   "question. Each should be answerable by reading the repo with "
                                   "read-only tools (read_file, list_dir, grep, glob).",
                },
            },
            "required": ["questions"],
        },
    },
}


def _tool_arguments(fn: dict) -> dict:
    """Decode a tool call's arguments.

    On the wire these are a JSON *string*, not an object. Both explore stages
    handed that string straight to execute_tool, which immediately did
    `.get(...)` on it - so every read_file, list_dir, grep and glob in the
    read-only fan-out returned `error: 'str' object has no attribute 'get'`.
    The agents never saw a single byte of the repository they were sent to
    read, and answered from priors instead, fluently enough that the failure
    only showed up in the event log.
    """
    raw = fn.get("arguments")
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("could not decode tool arguments for %s: %r", fn.get("name"), raw[:200])
        return {}
    return parsed if isinstance(parsed, dict) else {}


# -- stage 1: split -----------------------------------------------------

SYSTEM_PROMPT_SPLIT = (
    "You are exploring a repository to answer a single question. Your job is to decompose "
    "the question into a small set of independent sub-questions that can each be answered by "
    "reading the repo with read-only tools (read_file, list_dir, grep, glob).\n\n"
    "First, use list_dir to get an overview of the repo structure. Then use read_file and "
    "grep as needed to understand the codebase well enough to produce useful sub-questions.\n\n"
    "When you have your sub-questions, call submit_questions. Produce between 1 and 6 "
    "sub-questions. If the original question is simple and can be answered by investigating "
    "just one thing in the repo, submit a single sub-question — do not pad to hit a number.\n\n"
    "Sub-questions should be independent: answering one should not require knowing the answer "
    "to another. Group related concerns together rather than splitting hairs."
)

# The prompt above opens with "First, use list_dir..." - which on the final turn
# is the exact opposite of what is wanted, and beats tool_choice. ds4 offered
# ONLY submit_questions, with tool_choice forcing it, still answered with a
# fabricated list_dir call (inventing a `dir_path` argument the schema does not
# have) because the system prompt told it to explore first. Swapping the system
# message on the last turn is what actually makes forcing work.
SYSTEM_PROMPT_SPLIT_FINAL = (
    "You are decomposing a question about a repository into 1-6 independent sub-questions, "
    "each of which can be answered by reading the repo. Call submit_questions now with the "
    "best sub-questions you can form from what you already know. You have no other tools "
    "and no further turns."
)


async def _stage_split(
    client: OrchestratorClient,
    model: str,
    repo: Path,
    question: str,
    events: EventLog | None = None,
    max_turns: int = 10,
    max_tokens: int = 8192,
) -> tuple[list[str], int, int]:
    events = events or NullEventLog()
    """Run the explorer through the repo and collect sub-questions."""
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT_SPLIT},
        {"role": "user", "content": question},
    ]
    tools = READ_ONLY_TOOL_SCHEMAS + [SUBMIT_QUESTIONS_TOOL]

    split_prompt_tokens = 0
    split_completion_tokens = 0

    for turn in range(max_turns):
        forced_final = turn == max_turns - 1
        tool_choice = (
            {"type": "function", "function": {"name": "submit_questions"}}
            if forced_final else "auto"
        )
        # On the last turn, withdraw the exploration tools and the instruction to
        # use them; leaving either in place lets the model keep exploring right
        # past the deadline.
        turn_messages = messages
        turn_tools = tools
        if forced_final:
            turn_messages = [{"role": "system", "content": SYSTEM_PROMPT_SPLIT_FINAL}] + messages[1:]
            turn_tools = [SUBMIT_QUESTIONS_TOOL]
        try:
            completion = await client.chat_once(
                model, turn_messages, tools=turn_tools,
                tool_choice=tool_choice, max_tokens=max_tokens,
            )
        except OrchestratorError as e:
            events.emit("explore_split_error", turn=turn, status_code=e.status_code,
                        body=str(e.body))
            raise
        message = completion["choices"][0]["message"]
        usage = completion.get("usage") or {}
        split_prompt_tokens += int(usage.get("prompt_tokens") or 0)
        split_completion_tokens += int(usage.get("completion_tokens") or 0)
        events.emit_usage(
            "explore_split", model, usage,
            estimate=None if usage else estimate_usage(messages, message.get("content") or ""),
            turn=turn,
        )
        messages.append(message)

        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            messages.append({
                "role": "user",
                "content": "You must call submit_questions with your sub-questions, or continue "
                           "exploring with read_file/list_dir/grep/glob first if you need more "
                           "information.",
            })
            continue

        for call in tool_calls:
            fn = call["function"]
            name = fn["name"]
            if name == "submit_questions":
                questions = _tool_arguments(fn).get("questions", [])
                if not questions:
                    messages.append({
                        "role": "user",
                        "content": "submit_questions must include at least one sub-question. "
                                   "If you are not ready, continue exploring with read-only tools "
                                   "first.",
                    })
                    continue
                events.emit("explore_split_done", model=model, sub_question_count=len(questions))
                log.info("split: %d sub-question(s)", len(questions))
                return questions, split_prompt_tokens, split_completion_tokens

            # Execute the read-only tool and feed the result back.
            try:
                result = await execute_tool(name, _tool_arguments(fn), repo,
                                            run_shell_timeout_s=0)
            except Exception as e:
                result = f"error: {e}"
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": result,
            })

    # Ten turns of paid exploration have already happened; raising here throws
    # all of it away and returns nothing to the caller. The question itself is
    # always a valid sub-question, so degrade to a fan-out of one rather than
    # failing the run.
    log.warning("split: no submit_questions within %d turns; falling back to the question itself",
                max_turns)
    events.emit("explore_split_fallback", turns=max_turns, reason="no submit_questions call")
    return [question], split_prompt_tokens, split_completion_tokens


# -- stage 2: answer (per-sub-question agent loop) --------------------

SYSTEM_PROMPT_ANSWER = (
    "You are answering a specific sub-question about a repository. You have read-only "
    "access to the repo via read_file, list_dir, grep, and glob. Use these tools to "
    "find the information you need, then provide a clear, self-contained answer.\n\n"
    "When you have your answer, respond with plain text (no tool calls). If you cannot "
    "answer the question from what you can read, say so explicitly rather than guessing."
)


async def _run_single_agent(
    client: OrchestratorClient,
    *,
    model: str,
    repo: Path,
    sub_question: str,
    max_turns: int = 15,
    max_tokens: int = 8192,
    events: EventLog | None = None,
) -> SubQuestionResult:
    """Run one read-only agent on one sub-question. Never raises — errors are
    recorded in the result's error field."""
    result = SubQuestionResult(question=sub_question)
    events = events or NullEventLog()

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT_ANSWER},
        {"role": "user", "content": sub_question},
    ]

    events.emit("explore_agent_start", model=model, sub_question=sub_question[:200])

    try:
        for turn in range(max_turns):
            result.turns = turn + 1
            forced_final = turn == max_turns - 1
            tool_choice = (
                {"type": "none"}
                if forced_final else "auto"
            )
            try:
                completion = await client.chat_once(
                    model, messages, tools=READ_ONLY_TOOL_SCHEMAS,
                    tool_choice=tool_choice, max_tokens=max_tokens,
                )
            except OrchestratorError as e:
                events.emit("explore_agent_error", model=model, sub_question=sub_question[:200],
                            status_code=e.status_code, body=str(e.body))
                result.stop_reason = "error"
                result.error = f"orchestrator error {e.status_code}: {e.body}"
                return result

            message = completion["choices"][0]["message"]
            usage = completion.get("usage") or {}
            result.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            result.completion_tokens += int(usage.get("completion_tokens") or 0)
            events.emit_usage(
                "explore_agent", model, usage,
                estimate=None if usage else estimate_usage(messages, message.get("content") or ""),
                turn=turn, sub_question=sub_question[:100],
            )

            assistant: dict = {"role": "assistant", "content": message.get("content") or ""}
            if message.get("tool_calls"):
                assistant["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["function"]["name"], "arguments": json.dumps(tc["function"]["arguments"])},
                    }
                    for tc in message["tool_calls"]
                ]
            messages.append(assistant)

            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                # No more tool calls — the model gave us its answer.
                result.stop_reason = "finished"
                result.answer = message.get("content", "")
                result.ok = True
                break

            for tc in tool_calls:
                name = tc["function"]["name"]
                args = _tool_arguments(tc["function"])
                try:
                    output = await execute_tool(name, args, repo, run_shell_timeout_s=0)
                except Exception as e:
                    output = f"error: {e}"
                result.tool_calls += 1
                events.emit("tool_call", turn=turn, tool=name, sub_question=sub_question[:100],
                            result_preview=output[:200])
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": output,
                })
        else:
            result.stop_reason = "max_turns"
            result.error = f"hit the {max_turns}-turn ceiling"
    except asyncio.CancelledError:
        result.stop_reason = "cancelled"
        raise
    except Exception as e:
        result.stop_reason = "error"
        result.error = str(e)

    events.emit("explore_agent_end", model=model, sub_question=sub_question[:200],
                stop_reason=result.stop_reason, turns=result.turns,
                tool_calls=result.tool_calls, answer_preview=result.answer[:200] if result.answer else "")
    return result


# -- stage 3: synthesize ------------------------------------------------

SYSTEM_PROMPT_SYNTHESIZE = (
    "You are synthesizing the answers to a set of sub-questions into a single coherent "
    "response to the original question.\n\n"
    "For each sub-question, you have either an answer (possibly citing specific files) or "
    "a note that it went unanswered due to failure or other issue.\n\n"
    "Rules:\n"
    "1. Attribute claims to the files that support them wherever possible (e.g. "
    "\"src/router.py lines 10-25 selects the backend based on...\").\n"
    "2. If a sub-question went unanswered, say so plainly rather than papering over it.\n"
    "3. If the answers to the sub-questions collectively resolve the original question, "
    "state that clearly in your final paragraph.\n"
    "4. Do not invent information that is not supported by the sub-answers."
)


async def _stage_synthesize(
    client: OrchestratorClient,
    model: str,
    question: str,
    sub_answers: list[SubQuestionResult],
    events: EventLog | None = None,
    max_tokens: int = 8192,
) -> tuple[str, int, int]:
    """Synthesize a final answer from the sub-question results."""
    events = events or NullEventLog()

    # Build the synthesis prompt.
    prompt_parts = [
        f"Original question: {question}\n",
        "Sub-questions and answers:\n",
    ]
    for i, sa in enumerate(sub_answers, 1):
        if sa.answer:
            prompt_parts.append(f"\n{i}. {sa.question}\n   Answer: {sa.answer}\n")
        else:
            reason = sa.error or sa.stop_reason or "no answer produced"
            prompt_parts.append(f"\n{i}. {sa.question}\n   Status: UNANSWERED ({reason})\n")

    prompt_parts.append("\nProvide a single synthesized answer to the original question. "
                        "Attribute claims to files where possible. Note any gaps where a "
                        "sub-question was unanswered.")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_SYNTHESIZE},
        {"role": "user", "content": "\n".join(prompt_parts)},
    ]

    syn_prompt_tokens = 0
    syn_completion_tokens = 0

    try:
        completion = await client.chat_once(
            model, messages, tools=None,
            max_tokens=max_tokens,
        )
    except OrchestratorError as e:
        events.emit("explore_synthesize_error", status_code=e.status_code, body=str(e.body))
        raise

    answer = completion["choices"][0]["message"].get("content", "")
    usage = completion.get("usage") or {}
    syn_prompt_tokens += int(usage.get("prompt_tokens") or 0)
    syn_completion_tokens += int(usage.get("completion_tokens") or 0)
    events.emit_usage(
        "explore_synthesize", model, usage,
        estimate=None if usage else estimate_usage(messages, answer),
    )

    events.emit("explore_synthesize_done", sub_question_count=len(sub_answers),
                answer_chars=len(answer))
    return answer, syn_prompt_tokens, syn_completion_tokens


# -- top-level driver ---------------------------------------------------

async def run_explore(
    client: OrchestratorClient,
    *,
    model: str,
    repo: Path,
    question: str,
    max_questions: int = 6,
    agent_max_turns: int = 15,
    max_tokens: int = 8192,
    pipeline_cfg: PipelineCfg | None = None,
    events: EventLog | None = None,
) -> ExploreResult:
    """Run the full three-stage explore workflow.

    Returns an ExploreResult. Never raises for ordinary agent failure —
    sub-agent errors are recorded in the result so the synthesis stage
    can report gaps.
    """
    pipeline_cfg = pipeline_cfg or PipelineCfg()
    events = events or NullEventLog()
    started = time.monotonic()

    events.emit("explore_start", model=model, question=question[:500],
                repo=str(repo), max_questions=max_questions,
                agent_max_turns=agent_max_turns)

    # Stage 1: Split
    try:
        sub_questions, split_pt, split_ct = await _stage_split(
            client, model, repo, question, events, max_tokens=max_tokens,
        )
    except OrchestratorError:
        # Propagate orchestrator errors — they indicate a real problem,
        # not a recoverable agent failure.
        result = ExploreResult(question=question, sub_questions=[], sub_answers=[],
                               final_answer="", prompt_tokens=0, completion_tokens=0,
                               duration_s=time.monotonic() - started)
        events.emit("explore_end", ok=False, stop_reason="split_orchestrator_error")
        return result

    # Cap sub-questions at max_questions.
    sub_questions = sub_questions[:max_questions]

    # Stage 2: Answer each sub-question concurrently, bounded by
    # max_concurrent_workers.
    concurrency = pipeline_cfg.limits.max_concurrent_workers
    sub_answers = []
    all_tasks = [
        asyncio.create_task(
            _run_single_agent(
                client, model=model, repo=repo,
                sub_question=sub_questions[i],
                max_turns=agent_max_turns,
                max_tokens=max_tokens,
                events=events,
            ),
            name=f"explore-agent-{i}",
        )
        for i in range(len(sub_questions))
    ]
    events.emit("explore_agent_phase_start", sub_question_count=len(sub_questions),
                concurrency=concurrency)
    completed = await asyncio.gather(*all_tasks, return_exceptions=True)

    for i, result_or_exc in enumerate(zip(all_tasks, completed)):
        _, res = result_or_exc
        if isinstance(res, Exception):
            sub_answers.append(SubQuestionResult(
                question=sub_questions[i],
                stop_reason="error",
                error=f"agent crashed: {res}",
            ))
        else:
            sub_answers.append(res)

    # Stage 3: Synthesize
    try:
        final_answer, syn_pt, syn_ct = await _stage_synthesize(
            client, model, question, sub_answers, events,
        )
    except OrchestratorError:
        final_answer = (
            f"Synthesis failed. Original question: {question}\n\n"
            f"Sub-questions and partial results:\n"
        )
        for sa in sub_answers:
            final_answer += f"- {sa.question}: {sa.error or sa.stop_reason or 'no answer'}\n"
        syn_pt = 0
        syn_ct = 0

    total_prompt = split_pt + syn_pt + sum(a.prompt_tokens for a in sub_answers)
    total_completion = split_ct + syn_ct + sum(a.completion_tokens for a in sub_answers)
    duration = time.monotonic() - started

    events.emit("explore_end", ok=True, sub_question_count=len(sub_questions),
                answer_chars=len(final_answer), duration_s=round(duration, 1),
                prompt_tokens=total_prompt, completion_tokens=total_completion)

    return ExploreResult(
        question=question,
        sub_questions=sub_questions,
        sub_answers=sub_answers,
        final_answer=final_answer,
        prompt_tokens=total_prompt,
        completion_tokens=total_completion,
        duration_s=duration,
        split_prompt_tokens=split_pt,
        split_completion_tokens=split_ct,
        synthesize_prompt_tokens=syn_pt,
        synthesize_completion_tokens=syn_ct,
    )


async def explore_session(
    *,
    model: str,
    repo: Path,
    question: str,
    orchestrator_url: str,
    admin_url: str,
    api_key: str | None,
    pipeline_cfg: PipelineCfg,
    max_questions: int = 6,
    agent_max_turns: int = 15,
    max_tokens: int = 8192,
    events: EventLog | None = None,
    load_wait_s: float = 180.0,
    ensure_resident: bool = True,
) -> ExploreResult:
    """Wraps run_explore with client setup and residency for the explorer model."""
    async with OrchestratorClient(orchestrator_url, admin_url, api_key) as client:
        if ensure_resident:
            from .solo import ensure_model_resident
            await ensure_model_resident(client, model, load_wait_s)
        return await run_explore(
            client, model=model, repo=repo, question=question,
            max_questions=max_questions, agent_max_turns=agent_max_turns,
            max_tokens=max_tokens,
            pipeline_cfg=pipeline_cfg, events=events,
        )


__all__ = [
    "ExploreResult",
    "SubQuestionResult",
    "run_explore",
    "explore_session",
]
