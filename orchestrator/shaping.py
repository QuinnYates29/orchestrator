"""Per-backend request adaptation: make any OpenAI request safe and optimal
for the specific upstream it lands on."""
from __future__ import annotations

import json

from .config import ModelCfg
from .router import has_real_image, has_real_video

_SAMPLING_KEYS = ("temperature", "top_p", "presence_penalty",
                  "frequency_penalty", "logit_bias", "seed")
_MEDIA_TYPES = ("image_url", "input_image", "image", "video_url", "input_video", "video")


def _has_real_media_part(part: dict) -> bool:
    return has_real_image(part) or has_real_video(part)


def _strip_media_parts(messages: list, *, keep_real: bool) -> list:
    """Chat UIs commonly attach an empty image_url slot on every message
    whether or not the user picked an image. llama.cpp's server rejects the
    request outright if it structurally contains an image part and the model
    has no --mmproj, regardless of whether the url is actually empty — so
    these need to be dropped, not just excluded from routing decisions."""
    out = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            out.append(msg)
            continue
        kept = [
            p for p in content
            if not (
                isinstance(p, dict)
                and p.get("type") in _MEDIA_TYPES
                and (not keep_real or not _has_real_media_part(p))
            )
        ]
        out.append(msg if len(kept) == len(content) else {**msg, "content": kept})
    return out


def _has_real_media_message(msg: dict) -> bool:
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    return any(isinstance(part, dict) and part.get("type") in _MEDIA_TYPES and _has_real_media_part(part)
               for part in content)


def _trim_vision_context(messages: list) -> list:
    """Gemma is only the multimodal worker, so give it a small subprompt:
    latest system/developer instructions plus the most recent image/video
    message. This avoids loading the full conversation into the vision model."""
    system_prefix = [m for m in messages if m.get("role") in ("system", "developer")]
    media_msg = next((m for m in reversed(messages) if _has_real_media_message(m)), None)
    if media_msg is None:
        return messages
    return system_prefix + [media_msg]


def _content_as_text(content) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, separators=(",", ":"))


def _coalesce_system_messages(messages: list, *, extra_system: str | None = None) -> list:
    """llama.cpp/vLLM chat templates commonly require exactly one system
    message, and only as the first message. Collect system/developer content
    from anywhere in the conversation and emit it as a single leading system
    message."""
    system_parts: list[str] = []
    if extra_system:
        system_parts.append(extra_system)

    non_system = []
    for msg in messages:
        role = msg.get("role")
        if role in ("system", "developer"):
            content = msg.get("content")
            if content:
                system_parts.append(_content_as_text(content))
        else:
            non_system.append(msg)

    if not system_parts:
        return non_system
    return [{"role": "system", "content": "\n\n".join(system_parts)}] + non_system


def with_gemma_observation(body: dict, observation: str) -> dict:
    """Return a chat body for Ornith after Gemma has interpreted media."""
    out = dict(body)
    messages = out.get("messages")
    if not isinstance(messages, list):
        return out
    text_only = _strip_media_parts(messages, keep_real=False)
    text_only = _coalesce_system_messages(
        text_only,
        extra_system=(
            "Gemma was used only as the image/video perception worker for this "
            "request. Use the following observation as visual context; answer "
            "the user normally with your own reasoning and do not claim you "
            f"personally viewed the media.\n\nGemma observation:\n{observation}"
        ),
    )
    out["messages"] = text_only
    return out


def shape(body: dict, m: ModelCfg) -> tuple[dict, bool]:
    """Returns (upstream_body, synthesize_stream).

    synthesize_stream is True when the client asked to stream but the upstream
    can't (shipinabottle 400s on stream=true): we make a non-streaming upstream
    call and re-emit it as SSE so streaming clients still work.
    """
    out = dict(body)
    client_wants_stream = bool(out.get("stream"))

    if isinstance(out.get("messages"), list):
        if m.vision:
            messages = _strip_media_parts(out["messages"], keep_real=True)
            messages = _trim_vision_context(messages)
        else:
            messages = _strip_media_parts(out["messages"], keep_real=False)
        out["messages"] = _coalesce_system_messages(messages, extra_system=m.system_prompt)

    if m.upstream_model:
        out["model"] = m.upstream_model

    if m.greedy_only:
        for k in _SAMPLING_KEYS:
            out.pop(k, None)
    else:
        for k, v in m.sampling_defaults.items():
            out.setdefault(k, v)

    # Upstreams with tiny defaults (shipinabottle: 16) truncate everything
    # unless max_tokens is explicit.
    if m.force_max_tokens and not out.get("max_tokens") and not out.get("max_completion_tokens"):
        out["max_tokens"] = m.force_max_tokens

    # Always wins, even over what the client sent — for locking in behavior
    # (e.g. deterministic tool-calling) rather than just filling in a gap.
    for k, v in m.forced_params.items():
        out[k] = v

    synthesize = client_wants_stream and not m.streaming
    if synthesize:
        out["stream"] = False
        out.pop("stream_options", None)

    return out, synthesize
