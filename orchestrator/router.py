"""Prompt -> model routing. Pure functions, microsecond-cheap by design:
routing must never add meaningful latency on top of models that decode in
seconds per token."""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from .config import Cfg, RoutingCfg


@dataclass
class Route:
    preferred: str   # config model name the heuristics picked
    reason: str
    forced: bool = False  # client pinned a real model name; never fall back silently


@lru_cache(maxsize=64)
def _compile(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE)


def has_real_image(part: dict) -> bool:
    """Many chat UIs always structurally attach an image_url part (an empty
    attachment slot) whether or not the user picked an image — so presence of
    the key alone isn't image content, an actual non-empty url/data is."""
    ptype = part.get("type", "")
    if ptype in ("image_url", "input_image"):
        val = part.get("image_url") or part.get("image")
        url = val.get("url") if isinstance(val, dict) else val
        return bool(url and str(url).strip())
    if ptype == "image":
        source = part.get("source")
        if isinstance(source, dict):
            data = source.get("data") or source.get("url")
            return bool(data and str(data).strip())
        return bool(part.get("url"))
    return False


def has_real_video(part: dict) -> bool:
    """Return True only for non-empty video content, matching the loose shape
    of the common OpenAI-style multimodal content-part variants."""
    ptype = part.get("type", "")
    if ptype in ("video_url", "input_video"):
        val = part.get("video_url") or part.get("video")
        url = val.get("url") if isinstance(val, dict) else val
        return bool(url and str(url).strip())
    if ptype == "video":
        source = part.get("source")
        if isinstance(source, dict):
            data = source.get("data") or source.get("url")
            return bool(data and str(data).strip())
        return bool(part.get("url"))
    return False


def _message_has_real_media(msg: dict) -> bool:
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type", "")
        if ptype in ("image_url", "input_image", "image") and has_real_image(part):
            return True
        if ptype in ("video_url", "input_video", "video") and has_real_video(part):
            return True
    return False


def _latest_user_message(messages: list) -> dict | None:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg
    return None


def _iter_content_parts(messages: list) -> tuple[str, bool]:
    """Flatten chat messages to (text, has_visual_media).

    Text is gathered across the whole conversation (used for the precise/
    length heuristics below), but has_visual_media only looks at the latest
    user turn. Checking the full history means an image shared once early in
    a multi-turn conversation keeps re-triggering the Gemma subagent call on
    every later, unrelated text-only follow-up — a normal chat UI resends
    full history each turn, so this isn't a rare edge case."""
    texts: list[str] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in ("text", "input_text"):
                    texts.append(part.get("text", ""))

    latest_user = _latest_user_message(messages)
    has_visual_media = _message_has_real_media(latest_user) if latest_user else False

    return "\n".join(texts), has_visual_media


def _is_precise(text: str, rc: RoutingCfg) -> bool:
    return any(_compile(p).search(text) for p in rc.precise_patterns)


def _wants_escalation(body: dict, text: str, rc: RoutingCfg) -> tuple[bool, str]:
    """Three independent escalation signals, cheapest/least-ambiguous first.
    precise_patterns alone is narrowly math/proof-themed and misses "hard"
    prompts that aren't phrased that way — length and an explicit client
    signal catch what regex can't."""
    effort = str(body.get("reasoning_effort") or "").lower()
    if effort in ("high", "medium"):
        return True, "reasoning-effort"
    if len(text) >= rc.precise_min_chars:
        return True, "long-prompt"
    if _is_precise(text, rc):
        return True, "precise"
    return False, ""


def decide_chat(body: dict, cfg: Cfg) -> Route:
    rc = cfg.routing
    requested = str(body.get("model") or "auto")

    # Explicit pin to a real backend name: honor it, no heuristics, no fallback.
    if requested in cfg.models:
        return Route(requested, "client-pinned", forced=True)
    alias_target = cfg.aliases.get(requested)
    if alias_target and alias_target != "auto":
        return Route(alias_target, f"alias:{requested}")
    # Unknown names (incl. gpt-*) and "auto" fall through to heuristics.

    messages = body.get("messages") or []
    text, has_visual_media = _iter_content_parts(messages)

    if has_visual_media:
        if rc.vision:
            return Route(rc.default, "visual-media")
        return Route(rc.default, "vision-unconfigured")

    tools = body.get("tools") or ([body["functions"]] if body.get("functions") else [])
    resp_format = (body.get("response_format") or {}).get("type", "")
    if tools or resp_format == "json_schema":
        in_agent_loop = any(m.get("role") == "tool" for m in messages)
        complex_ = (
            len(tools) >= rc.tools_complex_min_tools
            or len(text) >= rc.tools_complex_min_chars
            or in_agent_loop
        )
        if complex_:
            return Route(rc.tools_complex, "tools-complex")
        return Route(rc.tools_simple, "tools-simple")

    escalate, reason = _wants_escalation(body, text, rc)
    if escalate:
        return Route(rc.precise, reason)

    return Route(rc.default, "default")


def decide_completion(body: dict, cfg: Cfg) -> Route:
    """Legacy /v1/completions: text-only, no tools/vision dimensions."""
    rc = cfg.routing
    requested = str(body.get("model") or "auto")
    if requested in cfg.models:
        return Route(requested, "client-pinned", forced=True)
    alias_target = cfg.aliases.get(requested)
    if alias_target and alias_target != "auto":
        return Route(alias_target, f"alias:{requested}")

    prompt = body.get("prompt") or ""
    if isinstance(prompt, list):
        prompt = "\n".join(p for p in prompt if isinstance(p, str))
    escalate, reason = _wants_escalation(body, str(prompt), rc)
    if escalate:
        return Route(rc.precise, reason)
    return Route(rc.default, "default")


def resolve(route: Route, cfg: Cfg, resident: set[str]) -> tuple[str | None, Route]:
    """Map preferred model to an actually-resident one via its fallback chain.
    Returns (chosen_or_None, route). Forced routes never redirect: a pinned
    model that is down is an error the client should see."""
    if route.forced:
        return (route.preferred if route.preferred in resident else None), route
    chain = [route.preferred] + list(cfg.models[route.preferred].fallback)
    for name in chain:
        if not cfg.models[name].enabled:
            continue
        if name in resident:
            return name, route
    return None, route


def serving_candidates(route: Route, cfg: Cfg) -> list[str]:
    """Ordered models that may legitimately serve this route — the preferred
    model plus (for non-forced routes) its enabled fallbacks. Used both for
    autoload and for retrying on a mid-request connection failure."""
    if route.forced:
        return [route.preferred]
    chain = [route.preferred] + list(cfg.models[route.preferred].fallback)
    seen: list[str] = []
    for name in chain:
        if cfg.models[name].enabled and name not in seen:
            seen.append(name)
    return seen


def request_text_chars(body: dict) -> int:
    """Flattened text length of a chat or legacy-completion body. Logged with
    each routing decision so escalation thresholds (precise_min_chars etc.)
    can be tuned against real traffic instead of guesses."""
    if isinstance(body.get("messages"), list):
        text, _ = _iter_content_parts(body["messages"])
        return len(text)
    prompt = body.get("prompt") or ""
    if isinstance(prompt, list):
        prompt = "\n".join(p for p in prompt if isinstance(p, str))
    return len(str(prompt))


def embedding_candidates(body: dict, cfg: Cfg) -> tuple[list[str], str]:
    """Models eligible to serve /v1/embeddings, in preference order.
    A pinned real model name (or alias) yields just that model — empty if it
    isn't embeddings-capable, which the endpoint turns into a clear 400."""
    requested = str(body.get("model") or "auto")
    target = cfg.aliases.get(requested, requested)
    if target in cfg.models:
        m = cfg.models[target]
        return ([target] if m.embeddings else []), "client-pinned"
    return [n for n, m in cfg.models.items() if m.enabled and m.embeddings], "embeddings-auto"
