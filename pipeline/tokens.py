"""Fallback token estimation for backends that report no usage.

Every backend in the current fleet reports usage once `stream_options.
include_usage` is set (see client._build_body), so this should not normally
fire. It exists because a run whose token cost silently drops to zero is worse
than one carrying an approximate number that is *labelled* approximate: the
whole point of the event log's token accounting is comparing two runs, and a
gap in the middle of that comparison is indistinguishable from a cheap run.

The estimate is a byte-per-token ratio, not a tokenizer. There is no tokenizer
library in this venv and adding one to guess at a *fallback* path is not worth
the dependency - the numbers here are only ever compared against each other,
so a consistent bias matters far less than being wrong in a hidden way. Events
produced from these numbers carry `estimated: true` and `reported: false`.
"""
from __future__ import annotations

import json

# Measured against the two fleet models on this project's own traffic (code,
# diffs, tool-call JSON), which runs denser than prose. 3.6 bytes/token was the
# median across ds4-full and ornith; prose-tuned rules of thumb (~4) undercount
# here.
BYTES_PER_TOKEN = 3.6

# Every message carries role/delimiter framing the content itself doesn't show.
PER_MESSAGE_OVERHEAD_TOKENS = 4


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, round(len(text.encode("utf-8")) / BYTES_PER_TOKEN))


def estimate_message_tokens(messages: list[dict]) -> int:
    """Approximate prompt size for a chat request, including tool-call payloads
    (which for a coding agent are frequently larger than the prose around
    them)."""
    total = 0
    for msg in messages:
        total += PER_MESSAGE_OVERHEAD_TOKENS
        content = msg.get("content")
        if isinstance(content, str):
            total += estimate_text_tokens(content)
        elif content:
            total += estimate_text_tokens(json.dumps(content, default=str))
        for call in msg.get("tool_calls") or []:
            total += estimate_text_tokens(json.dumps(call, default=str))
    return total


def estimate_usage(messages: list[dict], completion_text: str = "",
                   reasoning_chars: int = 0) -> dict:
    """A usage-shaped dict for a call whose backend reported nothing.

    `reasoning_chars` is counted separately because reasoning_content is
    generated and billed but never lands in `completion_text` - on ornith it is
    routinely an order of magnitude larger than the visible answer, so omitting
    it would understate a heavy reasoner by roughly that factor.
    """
    completion = estimate_text_tokens(completion_text)
    if reasoning_chars:
        completion += max(1, round(reasoning_chars / BYTES_PER_TOKEN))
    return {
        "prompt_tokens": estimate_message_tokens(messages),
        "completion_tokens": completion,
    }
