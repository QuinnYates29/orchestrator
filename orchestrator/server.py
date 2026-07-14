"""OpenAI-compatible orchestrator server: route -> shape -> relay."""
from __future__ import annotations

import argparse
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from . import router as routing
from . import shaping
from .backends import Registry
from .config import Cfg, load
from .stats import Stats

log = logging.getLogger("orchestrator")
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"

def _error(status: int, message: str, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": "orchestrator_error", "code": code}},
    )


def _chunk_sse(data: dict) -> bytes:
    return b"data: " + json.dumps(data, separators=(",", ":")).encode() + b"\n\n"


def _synthesized_sse(completion: dict, model: str, chunk_chars: int = 120):
    """Re-emit a full (non-streaming) chat completion as OpenAI SSE chunks."""
    cid = completion.get("id") or f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = completion.get("created") or int(time.time())
    base = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model}
    choice = (completion.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content") or ""
    finish = choice.get("finish_reason") or "stop"

    yield _chunk_sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
    for i in range(0, len(content), chunk_chars):
        yield _chunk_sse({**base, "choices": [{"index": 0, "delta": {"content": content[i:i + chunk_chars]}, "finish_reason": None}]})
    tail = {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}
    if completion.get("usage"):
        tail["usage"] = completion["usage"]
    yield _chunk_sse(tail)
    yield b"data: [DONE]\n\n"


def _completion_text(completion: dict) -> str:
    choice = (completion.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    text = choice.get("text")
    return text if isinstance(text, str) else ""


def _gemma_media_body(body: dict) -> dict:
    # gemma's persona/instructions come from its own config.yaml system_prompt,
    # applied uniformly by shaping.shape() for every model - no need to inject
    # anything here.
    out = dict(body)
    out["model"] = "gemma"
    out["stream"] = False
    out.pop("stream_options", None)
    return out


def create_app(cfg: Cfg) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # One shared pooled client: connection reuse is most of the proxy's speed.
        app.state.client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=32)
        )
        app.state.registry = Registry(cfg, app.state.client)
        await app.state.registry.start()
        yield
        await app.state.registry.stop()
        await app.state.client.aclose()

    app = FastAPI(title="model-orchestrator", lifespan=lifespan)
    app.state.stats = Stats()

    if cfg.api_key:
        @app.middleware("http")
        async def check_auth(request: Request, call_next):
            # /health stays open for load balancers / liveness probes.
            if request.url.path != "/health" and request.method != "OPTIONS":
                if request.headers.get("authorization", "") != f"Bearer {cfg.api_key}":
                    return _error(401, "missing or invalid API key", "invalid_api_key")
            return await call_next(request)

    if cfg.cors:
        # Added after the auth middleware so CORS wraps it and 401s still
        # carry CORS headers; expose_headers lets browser JS read the
        # x-orchestrator-* routing metadata.
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
            expose_headers=["*"],
        )

    async def _relay(request: Request, endpoint: str, decide):
        try:
            body = await request.json()
        except Exception:
            return _error(400, "request body must be JSON", "invalid_request")

        registry: Registry = app.state.registry
        client: httpx.AsyncClient = app.state.client

        route = decide(body, cfg)
        chosen, route = routing.resolve(route, cfg, registry.resident)

        stats: Stats = app.state.stats
        t0 = time.monotonic()
        text_chars = routing.request_text_chars(body)

        def _record(model: str, status: int, *, autoloaded: bool = False, usage: dict | None = None):
            stats.record(
                endpoint=endpoint, model=model, reason=route.reason, status=status,
                latency_ms=(time.monotonic() - t0) * 1000, preferred=route.preferred,
                autoloaded=autoloaded, usage=usage, text_chars=text_chars,
            )

        # Which models may serve this, in order. The resolved resident choice
        # (if any) goes first; the rest are retry candidates for connect
        # failures and autoload targets when nothing is up yet.
        candidates = routing.serving_candidates(route, cfg)
        if chosen is not None and chosen in candidates:
            candidates = [chosen] + [c for c in candidates if c != chosen]

        base_headers = {"x-orchestrator-reason": route.reason}

        if endpoint == "/chat/completions" and route.reason == "visual-media":
            gemma_name = cfg.routing.vision
            if not gemma_name:
                return _error(503, "visual media support is not configured", "vision_unconfigured")
            gemma_loaded = await registry.ensure_resident(gemma_name, LOG_DIR)
            if not gemma_loaded:
                return _error(
                    503,
                    f"visual media request needs {gemma_name}, but it is not resident and autoload failed",
                    "vision_model_not_resident",
                )
            gemma_cfg = cfg.models[gemma_name]
            gemma_body, _ = shaping.shape(_gemma_media_body(body), gemma_cfg)
            try:
                async with registry.active_request(gemma_name):
                    r = await client.post(
                        f"{gemma_cfg.upstream.rstrip('/')}/chat/completions",
                        json=gemma_body,
                        timeout=httpx.Timeout(gemma_cfg.timeout_s, connect=10.0),
                    )
            except httpx.ConnectError:
                registry.resident.discard(gemma_name)
                return _error(502, f"backend {gemma_name} refused connection at {gemma_cfg.upstream}", "backend_down")
            except httpx.TimeoutException:
                return _error(504, f"backend {gemma_name} timed out after {gemma_cfg.timeout_s}s", "backend_timeout")
            if r.status_code != 200:
                try:
                    data = r.json()
                except Exception:
                    data = {"error": {"message": r.text, "type": "backend_error", "code": "vision_backend_error"}}
                _record(gemma_name, r.status_code)
                return JSONResponse(status_code=r.status_code, content=data, headers=base_headers)
            body = shaping.with_gemma_observation(body, _completion_text(r.json()))
            base_headers["x-orchestrator-vision-model"] = gemma_name
            if gemma_cfg.unload_after_request:
                try:
                    registry.terminate(gemma_name)
                    base_headers["x-orchestrator-vision-unloaded"] = "true"
                except ValueError:
                    # Externally-started Gemma instances are intentionally not
                    # killed by the orchestrator.
                    pass

        async def attempt(name: str, headers: dict, *, autoloaded: bool):
            """One upstream try. Returns a response, or None for connect
            failure (caller moves on to the next candidate)."""
            m = cfg.models[name]
            upstream_body, synthesize = shaping.shape(body, m)
            url = f"{m.upstream.rstrip('/')}{endpoint}"
            timeout = httpx.Timeout(m.timeout_s, connect=10.0)

            active_cm = registry.active_request(name)
            await active_cm.__aenter__()
            active_open = True

            async def release_active():
                nonlocal active_open
                if active_open:
                    active_open = False
                    await active_cm.__aexit__(None, None, None)

            try:
                if synthesize:
                    r = await client.post(url, json=upstream_body, timeout=timeout)
                    await release_active()
                    completion = r.json()
                    _record(name, r.status_code, autoloaded=autoloaded,
                            usage=completion.get("usage") if isinstance(completion, dict) else None)
                    if r.status_code != 200:
                        return JSONResponse(status_code=r.status_code, content=completion, headers=headers)
                    return StreamingResponse(
                        _synthesized_sse(completion, name), media_type="text/event-stream", headers=headers
                    )

                req = client.build_request("POST", url, json=upstream_body, timeout=timeout)
                r = await client.send(req, stream=True)
                if r.status_code != 200 or not upstream_body.get("stream"):
                    content = await r.aread()
                    await r.aclose()
                    await release_active()
                    data = json.loads(content) if content else None
                    # Backends echo their own idea of "model" (llama.cpp: the raw
                    # .gguf path) — clients read this field, so it should name
                    # what actually served the request, not upstream internals.
                    if r.status_code == 200 and isinstance(data, dict) and "model" in data:
                        data["model"] = name
                    _record(name, r.status_code, autoloaded=autoloaded,
                            usage=data.get("usage") if isinstance(data, dict) else None)
                    return JSONResponse(status_code=r.status_code, content=data, headers=headers)

                async def passthrough():
                    usage = None
                    try:
                        async for line in r.aiter_lines():
                            if line.startswith("data: ") and line[6:].strip() != "[DONE]":
                                try:
                                    obj = json.loads(line[6:])
                                    obj["model"] = name
                                    if isinstance(obj.get("usage"), dict):
                                        usage = obj["usage"]
                                    line = "data: " + json.dumps(obj, separators=(",", ":"))
                                except json.JSONDecodeError:
                                    pass
                            if line:
                                yield (line + "\n\n").encode()
                    finally:
                        await r.aclose()
                        await release_active()
                        _record(name, 200, autoloaded=autoloaded, usage=usage)

                return StreamingResponse(
                    passthrough(),
                    media_type=r.headers.get("content-type", "text/event-stream"),
                    headers=headers,
                )
            except httpx.ConnectError:
                await release_active()
                registry.resident.discard(name)
                _record(name, 502, autoloaded=autoloaded)
                return None  # try the next candidate
            except httpx.TimeoutException:
                await release_active()
                _record(name, 504, autoloaded=autoloaded)
                return _error(504, f"backend {name} timed out after {m.timeout_s}s", "backend_timeout")
            except Exception:
                await release_active()
                raise

        tried: list[str] = []
        for name in candidates:
            autoloaded = False
            if name not in registry.resident:
                if not await registry.ensure_resident(name, LOG_DIR):
                    continue
                autoloaded = True
            headers = dict(base_headers)
            headers["x-orchestrator-model"] = name
            if name != route.preferred:
                headers["x-orchestrator-preferred"] = route.preferred
            if autoloaded:
                headers["x-orchestrator-autoloaded"] = "true"
            if tried:
                headers["x-orchestrator-retried-after"] = ",".join(tried)
            response = await attempt(name, headers, autoloaded=autoloaded)
            if response is not None:
                return response
            tried.append(name)

        if tried:
            return _error(
                502,
                f"all candidate backends refused connection: {', '.join(tried)}",
                "backend_down",
            )
        return _error(
            503,
            f"no backend available for this request (preferred: {route.preferred}, "
            f"reason: {route.reason}) — not resident and autoload didn't bring one up "
            f"in time (or none in the chain has launch_cmd/autoload set). "
            f"Check /admin/status and load one manually via POST /admin/load.",
            "model_not_resident",
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await _relay(request, "/chat/completions", routing.decide_chat)

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await _relay(request, "/completions", routing.decide_completion)

    @app.post("/v1/embeddings")
    async def embeddings(request: Request):
        try:
            body = await request.json()
        except Exception:
            return _error(400, "request body must be JSON", "invalid_request")

        registry: Registry = app.state.registry
        client: httpx.AsyncClient = app.state.client
        stats: Stats = app.state.stats
        t0 = time.monotonic()

        candidates, reason = routing.embedding_candidates(body, cfg)
        if not candidates:
            return _error(
                400,
                "no embeddings-capable model matches this request — pin a model with "
                "`embeddings: true` in config.yaml (llama-server backends also need "
                "--embeddings in their launch_cmd)",
                "no_embeddings_model",
            )
        # Prefer whatever is already warm; only autoload if nothing is.
        candidates.sort(key=lambda n: n not in registry.resident)

        for name in candidates:
            autoloaded = False
            if name not in registry.resident:
                if not await registry.ensure_resident(name, LOG_DIR):
                    continue
                autoloaded = True
            m = cfg.models[name]
            upstream_body = dict(body)
            if m.upstream_model:
                upstream_body["model"] = m.upstream_model
            try:
                async with registry.active_request(name):
                    r = await client.post(
                        f"{m.upstream.rstrip('/')}/embeddings",
                        json=upstream_body,
                        timeout=httpx.Timeout(m.timeout_s, connect=10.0),
                    )
            except httpx.ConnectError:
                registry.resident.discard(name)
                continue
            except httpx.TimeoutException:
                return _error(504, f"backend {name} timed out after {m.timeout_s}s", "backend_timeout")

            try:
                data = r.json()
            except Exception:
                data = None
            if r.status_code == 200 and isinstance(data, dict) and "model" in data:
                data["model"] = name
            stats.record(
                endpoint="/embeddings", model=name, reason=reason, status=r.status_code,
                latency_ms=(time.monotonic() - t0) * 1000, autoloaded=autoloaded,
                usage=data.get("usage") if isinstance(data, dict) else None,
            )
            headers = {"x-orchestrator-model": name, "x-orchestrator-reason": reason}
            if autoloaded:
                headers["x-orchestrator-autoloaded"] = "true"
            return JSONResponse(status_code=r.status_code, content=data, headers=headers)

        return _error(503, "no embeddings-capable backend could be brought up", "model_not_resident")

    @app.get("/v1/models")
    async def models():
        registry: Registry = app.state.registry
        now = int(time.time())
        data = [
            {"id": name, "object": "model", "created": now, "owned_by": "local",
             "resident": name in registry.resident, "tags": m.tags,
             "embeddings": m.embeddings, "vision": m.vision}
            for name, m in cfg.models.items()
        ]
        data += [
            {"id": alias, "object": "model", "created": now, "owned_by": "orchestrator",
             "alias_of": target}
            for alias, target in cfg.aliases.items()
        ]
        return {"object": "list", "data": data}

    @app.get("/v1/models/{model_id}")
    async def model_detail(model_id: str):
        registry: Registry = app.state.registry
        now = int(time.time())
        if model_id in cfg.models:
            m = cfg.models[model_id]
            return {"id": model_id, "object": "model", "created": now, "owned_by": "local",
                    "resident": model_id in registry.resident, "tags": m.tags,
                    "embeddings": m.embeddings, "vision": m.vision}
        if model_id in cfg.aliases:
            return {"id": model_id, "object": "model", "created": now,
                    "owned_by": "orchestrator", "alias_of": cfg.aliases[model_id]}
        return _error(404, f"model {model_id!r} not found", "model_not_found")

    @app.get("/health")
    async def health():
        return {"ok": True, "resident": sorted(app.state.registry.resident)}

    @app.get("/admin/status")
    async def admin_status():
        return app.state.registry.status()

    @app.get("/admin/stats")
    async def admin_stats():
        return app.state.stats.snapshot()

    @app.post("/admin/load")
    async def admin_load(request: Request):
        body = await request.json()
        name = body.get("model", "")
        wait_s = body.get("wait_s")
        if name not in cfg.models:
            return _error(404, f"unknown model {name!r}", "model_not_found")

        registry: Registry = app.state.registry
        if name in registry.resident:
            detail = f"{name} already resident"
        else:
            try:
                detail = await registry.launch_for_load(name, LOG_DIR)
            except ValueError as e:
                return _error(400, str(e), "launch_unavailable")

        if wait_s:
            ready = await registry.wait_resident(name, float(wait_s))
            if not ready:
                return _error(504, f"{name} not resident after {wait_s}s", "load_timeout")
            return {"ok": True, "resident": True, "detail": detail}
        return {"ok": True, "detail": detail}

    @app.post("/admin/unload")
    async def admin_unload(request: Request):
        body = await request.json()
        name = body.get("model", "")
        wait_idle_s = float(body.get("wait_idle_s", 0.0) or 0.0)
        force = bool(body.get("force", False))
        if name not in cfg.models:
            return _error(404, f"unknown model {name!r}", "model_not_found")
        try:
            if force:
                msg = app.state.registry.terminate(name)
            else:
                msg = await app.state.registry.unload_when_idle(name, wait_idle_s)
        except ValueError as e:
            return _error(400, str(e), "not_managed")
        return {"ok": True, "detail": msg}

    return app


def main():
    parser = argparse.ArgumentParser(description="OpenAI-compatible model orchestrator")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "config.yaml"))
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # silences one INFO line per health probe
    cfg = load(args.config)
    uvicorn.run(create_app(cfg), host=args.host or cfg.host, port=args.port or cfg.port)


if __name__ == "__main__":
    main()
