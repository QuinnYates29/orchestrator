from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from orchestrator import config, router, shaping
from orchestrator.backends import Registry

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


@pytest.fixture(scope="module")
def cfg() -> config.Cfg:
    return config.load(CONFIG_PATH)


def _chat(text: str, **extra) -> dict:
    body = {"model": "auto", "messages": [{"role": "user", "content": text}]}
    body.update(extra)
    return body


# -- decide_chat: default / content heuristics -------------------------------


def test_default_routes_to_ornith(cfg):
    route = router.decide_chat(_chat("what's the weather like"), cfg)
    assert route.preferred == "ornith"
    assert route.reason == "default"
    assert route.forced is False


def test_image_content_part_routes_to_gemma(cfg):
    body = {
        "model": "auto",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}},
                ],
            }
        ],
    }
    route = router.decide_chat(body, cfg)
    assert route.preferred == "ornith"
    assert route.reason == "visual-media"


def test_video_content_part_routes_to_gemma(cfg):
    body = {
        "model": "auto",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "summarize this video"},
                    {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,xxx"}},
                ],
            }
        ],
    }
    route = router.decide_chat(body, cfg)
    assert route.preferred == "ornith"
    assert route.reason == "visual-media"


def test_empty_image_url_part_does_not_route_to_gemma(cfg):
    # Many chat UIs always attach an image_url slot structurally, empty when
    # no image is picked. Presence of the key alone must not trigger vision
    # routing, or every request from such a UI lands on gemma regardless of
    # content.
    body = {
        "model": "auto",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what's the weather like"},
                    {"type": "image_url", "image_url": {"url": ""}},
                ],
            }
        ],
    }
    route = router.decide_chat(body, cfg)
    assert route.preferred == "ornith"
    assert route.reason == "default"


def test_stale_image_earlier_in_history_does_not_stick_to_gemma(cfg):
    # A normal multi-turn chat UI resends full history each turn. If an image
    # was shared once earlier in the conversation, a later unrelated
    # text-only follow-up must not keep re-triggering the Gemma subagent —
    # only the latest user turn should be checked for real media.
    body = {
        "model": "auto",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}},
                ],
            },
            {"role": "assistant", "content": "That's a red cube."},
            {"role": "user", "content": "thanks, and what's 2+2?"},
        ],
    }
    route = router.decide_chat(body, cfg)
    assert route.preferred == "ornith"
    assert route.reason == "default"


def test_latest_turn_with_real_image_still_routes_to_gemma(cfg):
    body = {
        "model": "auto",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}},
                ],
            },
        ],
    }
    route = router.decide_chat(body, cfg)
    assert route.preferred == "ornith"
    assert route.reason == "visual-media"


def test_long_prompt_escalates_without_pattern_match(cfg):
    # precise_patterns is narrowly math/proof-themed; a long prompt with none
    # of those phrases should still escalate on length alone.
    body = _chat("please help me. " * 200)
    route = router.decide_chat(body, cfg)
    assert route.preferred == "ds4"
    assert route.reason == "long-prompt"


def test_reasoning_effort_high_escalates_regardless_of_content(cfg):
    body = _chat("hi", reasoning_effort="high")
    route = router.decide_chat(body, cfg)
    assert route.preferred == "ds4"
    assert route.reason == "reasoning-effort"


def test_short_plain_prompt_does_not_escalate(cfg):
    route = router.decide_chat(_chat("what's the weather like"), cfg)
    assert route.preferred == "ornith"
    assert route.reason == "default"


def test_tools_simple_routes_to_ornith(cfg):
    body = _chat("what's 2+2", tools=[{"type": "function", "function": {"name": "add"}}])
    route = router.decide_chat(body, cfg)
    assert route.preferred == "ornith"
    assert route.reason == "tools-simple"


def test_tools_complex_by_count_routes_to_ds4(cfg):
    tools = [{"type": "function", "function": {"name": f"tool{i}"}} for i in range(3)]
    body = _chat("do the thing", tools=tools)
    route = router.decide_chat(body, cfg)
    assert route.preferred == "ds4"
    assert route.reason == "tools-complex"


def test_agent_loop_role_tool_routes_to_ds4(cfg):
    body = {
        "model": "auto",
        "tools": [{"type": "function", "function": {"name": "add"}}],
        "messages": [
            {"role": "user", "content": "add these numbers"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
            {"role": "tool", "content": "4", "tool_call_id": "1"},
        ],
    }
    route = router.decide_chat(body, cfg)
    assert route.preferred == "ds4"
    assert route.reason == "tools-complex"


def test_precise_pattern_routes_to_ds4(cfg):
    route = router.decide_chat(_chat("please prove the theorem for me"), cfg)
    assert route.preferred == "ds4"
    assert route.reason == "precise"


# -- decide_chat: explicit pin / alias / unknown -----------------------------


def test_explicit_model_name_is_forced_pin(cfg):
    route = router.decide_chat(_chat("hi", model="glm"), cfg)
    assert route.preferred == "glm"
    assert route.reason == "client-pinned"
    assert route.forced is True


def test_alias_orchestrator_auto_routes_by_content(cfg):
    route = router.decide_chat(_chat("hi", model="orchestrator"), cfg)
    assert route.preferred == "ornith"
    assert route.reason == "default"
    assert route.forced is False


def test_removed_mode_alias_auto_routes_by_content(cfg):
    route = router.decide_chat(_chat("please prove the theorem", model="precise"), cfg)
    assert route.preferred == "ds4"
    assert route.reason == "precise"
    assert route.forced is False


def test_unknown_model_name_auto_routes_by_content(cfg):
    route = router.decide_chat(_chat("please prove the theorem", model="gpt-4o"), cfg)
    assert route.preferred == "ds4"
    assert route.reason == "precise"
    assert route.forced is False


# -- resolve() -----------------------------------------------------------


def test_resolve_forced_not_resident_returns_none(cfg):
    route = router.Route(preferred="glm", reason="client-pinned", forced=True)
    chosen, out_route = router.resolve(route, cfg, resident=set())
    assert chosen is None
    assert out_route is route


def test_resolve_forced_resident_returns_itself(cfg):
    route = router.Route(preferred="glm", reason="client-pinned", forced=True)
    chosen, _ = router.resolve(route, cfg, resident={"glm"})
    assert chosen == "glm"


def test_resolve_nonforced_falls_back_through_chain(cfg):
    route = router.Route(preferred="ds4", reason="tools-complex", forced=False)
    chosen, _ = router.resolve(route, cfg, resident={"ornith"})
    assert chosen == "ornith"


def test_ds4_is_enabled_after_download(cfg):
    assert cfg.models["ds4"].enabled is True
    assert cfg.models["ds4"].autoload is True
    assert "ds4flash.gguf" in cfg.models["ds4"].launch_cmd
    assert cfg.models["ds4"].unload_before_load == ["ornith", "glm", "gemma"]


def test_heavy_model_load_unloads_managed_conflicts(cfg, monkeypatch, tmp_path):
    registry = Registry(cfg, client=None)
    registry.resident = {"ornith", "gemma"}
    registry.procs = {
        "ornith": SimpleNamespace(poll=lambda: None),
        "gemma": SimpleNamespace(poll=lambda: None),
    }
    called = []

    async def fake_probe(name):
        return None

    def fake_terminate(name):
        called.append(f"terminate:{name}")
        registry.resident.discard(name)
        return f"terminated {name}"

    def fake_launch(name, log_dir):
        called.append(f"launch:{name}")
        return f"launched {name}"

    monkeypatch.setattr(registry, "_probe", fake_probe)
    monkeypatch.setattr(registry, "terminate", fake_terminate)
    monkeypatch.setattr(registry, "launch", fake_launch)

    detail = asyncio.run(registry.launch_for_load("ds4", tmp_path))

    assert detail == "launched ds4"
    assert called == ["terminate:ornith", "terminate:gemma", "launch:ds4"]


def test_heavy_model_load_refuses_external_conflict(cfg, monkeypatch, tmp_path):
    registry = Registry(cfg, client=None)
    registry.resident = {"ornith"}

    async def fake_probe(name):
        return None

    monkeypatch.setattr(registry, "_probe", fake_probe)

    with pytest.raises(ValueError, match="ornith is resident"):
        asyncio.run(registry.launch_for_load("ds4", tmp_path))


def test_resolve_skips_disabled_model_in_chain(cfg):
    # glm is disabled in config.yaml; even if resident, resolve() must not
    # choose it non-forced, and must fall through to the next in chain.
    route = router.Route(preferred="glm", reason="precise", forced=False)
    chosen, _ = router.resolve(route, cfg, resident={"glm", "ornith"})
    assert chosen == "ornith"


def test_resolve_nonforced_none_resident_returns_none(cfg):
    route = router.Route(preferred="ornith", reason="default", forced=False)
    chosen, _ = router.resolve(route, cfg, resident=set())
    assert chosen is None


# -- serving_candidates / embedding_candidates / request_text_chars ----------


def test_serving_candidates_forced_is_single(cfg):
    route = router.Route(preferred="ds4", reason="client-pinned", forced=True)
    assert router.serving_candidates(route, cfg) == ["ds4"]


def test_serving_candidates_includes_enabled_fallbacks(cfg):
    route = router.Route(preferred="ds4", reason="tools-complex", forced=False)
    # ds4 fallback is [ornith]; both enabled.
    assert router.serving_candidates(route, cfg) == ["ds4", "ornith"]


def test_serving_candidates_skips_disabled(cfg):
    # glm is disabled; if it were the preferred non-forced target it drops out.
    route = router.Route(preferred="glm", reason="precise", forced=False)
    cands = router.serving_candidates(route, cfg)
    assert "glm" not in cands
    assert "ornith" in cands  # glm's fallback


def test_embedding_candidates_auto_picks_embeddings_models(cfg):
    cands, reason = router.embedding_candidates({"model": "auto"}, cfg)
    assert "ornith" in cands
    assert reason == "embeddings-auto"


def test_embedding_candidates_pinned_non_embeddings_is_empty(cfg):
    # gemma is not embeddings-capable; pinning it yields no candidate.
    cands, reason = router.embedding_candidates({"model": "gemma"}, cfg)
    assert cands == []
    assert reason == "client-pinned"


def test_request_text_chars_counts_chat_and_completion(cfg):
    chat = {"messages": [{"role": "user", "content": "hello world"}]}
    assert router.request_text_chars(chat) == len("hello world")
    completion = {"prompt": "abc"}
    assert router.request_text_chars(completion) == 3


# -- shaping ---------------------------------------------------------------


def test_shape_glm_strips_sampling_and_forces_max_tokens_and_synthesizes(cfg):
    body = {
        "model": "glm",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.9,
        "stream": True,
    }
    out, synthesize = shaping.shape(body, cfg.models["glm"])
    assert "temperature" not in out
    assert out["max_tokens"] == 512
    assert synthesize is True
    assert out["stream"] is False
    assert out["model"] == "glm-5.2"


def test_shape_glm_respects_explicit_max_tokens(cfg):
    body = {"model": "glm", "messages": [], "max_tokens": 64}
    out, _ = shaping.shape(body, cfg.models["glm"])
    assert out["max_tokens"] == 64


def test_shape_ornith_sets_default_temperature_when_unset(cfg):
    body = {"model": "ornith", "messages": [{"role": "user", "content": "hi"}]}
    out, synthesize = shaping.shape(body, cfg.models["ornith"])
    assert out["temperature"] == 0.7
    assert synthesize is False
    assert out["messages"][0]["role"] == "system"
    assert "default chat model and router" in out["messages"][0]["content"]


def test_shape_ornith_coalesces_late_system_messages_to_front(cfg):
    body = {
        "model": "ornith",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "late instruction"},
        ],
    }
    out, _ = shaping.shape(body, cfg.models["ornith"])
    assert [m["role"] for m in out["messages"]] == ["system", "user"]
    assert "default chat model and router" in out["messages"][0]["content"]
    assert "late instruction" in out["messages"][0]["content"]


def test_shape_ornith_preserves_explicit_temperature(cfg):
    body = {"model": "ornith", "messages": [], "temperature": 0.2}
    out, _ = shaping.shape(body, cfg.models["ornith"])
    assert out["temperature"] == 0.2


def test_shape_strips_empty_image_url_part(cfg):
    # llama.cpp's server 500s on a structurally-present image part when the
    # model has no --mmproj, regardless of whether the url is empty — must
    # be dropped before forwarding, not just excluded from routing.
    body = {
        "model": "ornith",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "hi"},
                {"type": "image_url", "image_url": {"url": ""}},
            ],
        }],
    }
    out, _ = shaping.shape(body, cfg.models["ornith"])
    parts = out["messages"][-1]["content"]
    assert [p["type"] for p in parts] == ["text"]


def test_shape_preserves_real_image_url_part(cfg):
    body = {
        "model": "gemma",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}},
            ],
        }],
    }
    out, _ = shaping.shape(body, cfg.models["gemma"])
    # gemma now has its own system_prompt (config.yaml), injected as a
    # leading message — the user message is no longer necessarily index 0.
    user_msg = next(m for m in out["messages"] if m["role"] == "user")
    parts = user_msg["content"]
    assert [p["type"] for p in parts] == ["text", "image_url"]
    assert parts[1]["image_url"]["url"] == "data:image/png;base64,xxx"


def test_shape_ornith_strips_real_media_parts(cfg):
    body = {
        "model": "ornith",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}},
            ],
        }],
    }
    out, _ = shaping.shape(body, cfg.models["ornith"])
    parts = out["messages"][-1]["content"]
    assert [p["type"] for p in parts] == ["text"]


def test_shape_gemma_trims_to_latest_media_subprompt(cfg):
    body = {
        "model": "gemma",
        "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "unrelated earlier context"},
            {"role": "assistant", "content": "old answer"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is shown here?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}},
                ],
            },
        ],
    }
    out, _ = shaping.shape(body, cfg.models["gemma"])
    assert len(out["messages"]) == 2
    assert out["messages"][0]["role"] == "system"
    assert out["messages"][1]["content"][0]["text"] == "what is shown here?"


def test_with_gemma_observation_keeps_observation_in_first_system_message(cfg):
    body = {
        "model": "ornith",
        "messages": [
            {"role": "system", "content": "existing instruction"},
            {"role": "user", "content": "what is this?"},
        ],
    }
    body = shaping.with_gemma_observation(body, "a red cube")
    out, _ = shaping.shape(body, cfg.models["ornith"])
    assert [m["role"] for m in out["messages"]] == ["system", "user"]
    assert "Gemma observation:\na red cube" in out["messages"][0]["content"]
    assert "existing instruction" in out["messages"][0]["content"]
