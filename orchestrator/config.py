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
    default_profile: str | None = None  # named launch profile used for autoload/admin loads
    enabled: bool = True         # excluded from auto-routing/fallback when False; direct pin still works
    autoload: bool = True        # launch_cmd fires automatically on first request that needs it
    load_timeout_s: float = 180.0
    unload_before_load: list[str] = field(default_factory=list)
    unload_after_request: bool = False
    restart: bool = True         # watchdog auto-relaunches this backend if it crashes while managed


@dataclass
class LaunchProfileCfg:
    """One concrete way to run a logical model.

    Profiles share the model's upstream/API behavior but may have radically
    different runtime tradeoffs, such as DS4's SSD-streaming versus fully
    resident weights.  They are selected only for lifecycle operations; route
    decisions continue to use the logical model name.
    """
    name: str
    model: str
    launch_cmd: str
    unload_before_load: list[str] = field(default_factory=list)
    description: str | None = None


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
    fail_threshold: int = 2            # consecutive failed probes before marking a backend down (debounce)
    restart_backoff_s: float = 10.0    # base delay before a crashed backend is auto-relaunched
    restart_max_backoff_s: float = 120.0  # cap on exponential restart backoff
    autoload_max_failures: int = 3     # consecutive failed autoloads before the circuit opens
    autoload_cooldown_s: float = 30.0  # how long the circuit stays open (fast-fail) after that


@dataclass
class Cfg:
    host: str
    port: int
    models: dict[str, ModelCfg]
    launch_profiles: dict[str, LaunchProfileCfg]
    aliases: dict[str, str]
    routing: RoutingCfg
    health: HealthCfg
    api_key: str | None = None   # when set, /v1/* and /admin/* require Authorization: Bearer <key>
    cors: bool = True            # allow browser clients (dashboards) to call the API

    def profile_for(self, model: str, profile: str | None = None) -> LaunchProfileCfg | None:
        """Resolve a requested profile, or the model's autoload default.

        ``None`` preserves legacy single-command model configurations.
        """
        m = self.models[model]
        selected = profile or m.default_profile
        if selected is None:
            return None
        p = self.launch_profiles.get(selected)
        if p is None:
            raise ValueError(f"unknown launch profile {selected!r}")
        if p.model != model:
            raise ValueError(f"launch profile {selected!r} belongs to {p.model!r}, not {model!r}")
        return p

    def profiles_for(self, model: str) -> list[str]:
        return [name for name, p in self.launch_profiles.items() if p.model == model]


def load(path: str | Path) -> Cfg:
    raw = yaml.safe_load(Path(path).read_text())

    models: dict[str, ModelCfg] = {}
    for name, m in (raw.get("models") or {}).items():
        m = dict(m or {})
        m.pop("name", None)
        models[name] = ModelCfg(name=name, **m)

    launch_profiles: dict[str, LaunchProfileCfg] = {}
    for name, p in (raw.get("launch_profiles") or {}).items():
        p = dict(p or {})
        p.pop("name", None)
        launch_profiles[name] = LaunchProfileCfg(name=name, **p)

    for name, p in launch_profiles.items():
        if p.model not in models:
            raise ValueError(f"launch profile {name!r} -> unknown model {p.model!r}")
        for other in p.unload_before_load:
            if other not in models:
                raise ValueError(f"launch profile {name!r} conflicts with unknown model {other!r}")
    for name, m in models.items():
        if m.default_profile:
            if m.default_profile not in launch_profiles:
                raise ValueError(f"{name}.default_profile -> unknown profile {m.default_profile!r}")
            if launch_profiles[m.default_profile].model != name:
                raise ValueError(f"{name}.default_profile {m.default_profile!r} belongs to a different model")

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
        launch_profiles=launch_profiles,
        aliases=aliases,
        routing=routing,
        health=HealthCfg(**(raw.get("health") or {})),
        api_key=raw.get("api_key") or None,
        cors=bool(raw.get("cors", True)),
    )
