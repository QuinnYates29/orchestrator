from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelCfg:
    name: str
    upstream: str
    upstream_model: str | None = None
    tags: list[str] = field(default_factory=list)
    streaming: bool = True
    greedy_only: bool = False
    vision: bool = False
    embeddings: bool = False     # backend can serve /v1/embeddings (llama-server needs --embeddings)
    timeout_s: float = 600.0
    force_max_tokens: int | None = None
    fallback: list[str] = field(default_factory=list)
    sampling_defaults: dict = field(default_factory=dict)   # applied only if client didn't set it
    forced_params: dict = field(default_factory=dict)       # always applied, overrides client input
    system_prompt: str | None = None                        # persistent per-model context/persona
    launch_cmd: str | None = None
    enabled: bool = True         # excluded from auto-routing/fallback when False; direct pin still works
    autoload: bool = True        # launch_cmd fires automatically on first request that needs it
    load_timeout_s: float = 180.0
    unload_before_load: list[str] = field(default_factory=list)
    unload_after_request: bool = False


@dataclass
class RoutingCfg:
    default: str
    vision: str | None
    tools_simple: str
    tools_complex: str
    precise: str
    precise_patterns: list[str] = field(default_factory=list)
    tools_complex_min_tools: int = 3
    tools_complex_min_chars: int = 4000
    precise_min_chars: int = 2500


@dataclass
class HealthCfg:
    interval_s: float = 5.0
    probe_timeout_s: float = 1.5


@dataclass
class Cfg:
    host: str
    port: int
    models: dict[str, ModelCfg]
    aliases: dict[str, str]
    routing: RoutingCfg
    health: HealthCfg
    api_key: str | None = None   # when set, /v1/* and /admin/* require Authorization: Bearer <key>
    cors: bool = True            # allow browser clients (dashboards) to call the API


def load(path: str | Path) -> Cfg:
    raw = yaml.safe_load(Path(path).read_text())

    models: dict[str, ModelCfg] = {}
    for name, m in (raw.get("models") or {}).items():
        m = dict(m or {})
        m.pop("name", None)
        models[name] = ModelCfg(name=name, **m)

    routing = RoutingCfg(**(raw.get("routing") or {}))
    for target in [routing.default, routing.tools_simple, routing.tools_complex,
                   routing.precise] + ([routing.vision] if routing.vision else []):
        if target not in models:
            raise ValueError(f"routing target {target!r} is not a configured model")

    aliases = {str(k): str(v) for k, v in (raw.get("aliases") or {}).items()}
    for alias, target in aliases.items():
        if target != "auto" and target not in models:
            raise ValueError(f"alias {alias!r} -> unknown model {target!r}")

    listen = raw.get("listen") or {}
    return Cfg(
        host=listen.get("host", "127.0.0.1"),
        port=int(listen.get("port", 8080)),
        models=models,
        aliases=aliases,
        routing=routing,
        health=HealthCfg(**(raw.get("health") or {})),
        api_key=raw.get("api_key") or None,
        cors=bool(raw.get("cors", True)),
    )
