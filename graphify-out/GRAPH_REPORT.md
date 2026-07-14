# Graph Report - /home/quinna/tools/orchestrator  (2026-07-13)

## Corpus Check
- Corpus is ~8,261 words - fits in a single context window. You may not need a graph.

## Summary
- 159 nodes · 246 edges · 13 communities (9 shown, 4 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 4 edges (avg confidence: 0.59)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Router Test Suite
- Backend Registry & Lifecycle
- Media Detection & Shaping
- Config & Routing Core
- Model Backend Configs
- Admin API & Routing Rationale
- HTTP Server & SSE
- Stub Test Backend
- GLM Streaming Synthesis
- Listen Config
- Orchestrator Root
- Model Aliases Table
- Response Headers

## God Nodes (most connected - your core abstractions)
1. `Registry` - 23 edges
2. `Cfg` - 12 edges
3. `_chat()` - 9 edges
4. `load()` - 8 edges
5. `Route` - 6 edges
6. `shape()` - 6 edges
7. `models.ds4 config entry` - 6 edges
8. `RoutingCfg` - 5 edges
9. `has_real_image()` - 5 edges
10. `has_real_video()` - 5 edges

## Surprising Connections (you probably didn't know these)
- `test_heavy_model_load_refuses_external_conflict()` --calls--> `Registry`  [EXTRACTED]
  tests/test_router.py → orchestrator/backends.py
- `test_heavy_model_load_unloads_managed_conflicts()` --calls--> `Registry`  [EXTRACTED]
  tests/test_router.py → orchestrator/backends.py
- `Routing decision logic` --references--> `routing config block`  [EXTRACTED]
  README.md → config.yaml
- `glm backend (shipinabottle GLM)` --references--> `models.glm config entry`  [EXTRACTED]
  README.md → config.yaml
- `Rationale: GLM disabled (sub-1 tok/s decode)` --references--> `models.glm config entry`  [EXTRACTED]
  README.md → config.yaml

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Configured model backend fleet (glm, ornith, ds4, gemma)** — config_glm, config_ornith, config_ds4, config_gemma [EXTRACTED 1.00]
- **route -> shape -> relay request pipeline** — readme_routing, readme_router_py, readme_shaping_py, readme_server_py [EXTRACTED 1.00]
- **Backend residency management (resident set, autoload, admin load/unload)** — readme_resident_set, readme_autoload, readme_admin_load, readme_admin_unload [INFERRED 0.85]

## Communities (13 total, 4 thin omitted)

### Community 0 - "Router Test Suite"
Cohesion: 0.08
Nodes (9): _chat(), test_alias_orchestrator_auto_routes_by_content(), test_default_routes_to_ornith(), test_explicit_model_name_is_forced_pin(), test_precise_pattern_routes_to_ds4(), test_removed_mode_alias_auto_routes_by_content(), test_tools_complex_by_count_routes_to_ds4(), test_tools_simple_routes_to_ornith() (+1 more)

### Community 1 - "Backend Registry & Lifecycle"
Cohesion: 0.15
Nodes (7): AsyncClient, Path, Poll until `name` shows resident or timeout_s elapses. For callers         (agen, First-request autoload: launch (idempotent) + block until ready,         seriali, Registry, test_heavy_model_load_refuses_external_conflict(), test_heavy_model_load_unloads_managed_conflicts()

### Community 2 - "Media Detection & Shaping"
Cohesion: 0.15
Nodes (20): has_real_image(), has_real_video(), _iter_content_parts(), Many chat UIs always structurally attach an image_url part (an empty     attachm, Return True only for non-empty video content, matching the loose shape     of th, Flatten chat messages to (text, has_visual_media)., _coalesce_system_messages(), _content_as_text() (+12 more)

### Community 3 - "Config & Routing Core"
Cohesion: 0.20
Nodes (17): Backend residency tracking and explicit-swap process management.  Residency is p, Cfg, HealthCfg, load(), ModelCfg, Path, RoutingCfg, _compile() (+9 more)

### Community 4 - "Model Backend Configs"
Cohesion: 0.17
Nodes (17): aliases mapping, models.ds4 config entry, models.gemma config entry, models.glm config entry, models.ornith config entry, routing config block, DeepSeek V4 Flash model, ds4 backend (DeepSeek V4 Flash) (+9 more)

### Community 5 - "Admin API & Routing Rationale"
Cohesion: 0.13
Nodes (17): health probe config, POST /admin/load endpoint, GET /admin/status endpoint, POST /admin/unload endpoint, Autoload (first-request cold start), ds4-server (antirez's DeepSeek V4 inference engine), router.has_real_image(), Rationale: orchestrator doesn't manage model loading itself (+9 more)

### Community 6 - "HTTP Server & SSE"
Cohesion: 0.18
Nodes (10): FastAPI, JSONResponse, OpenAI-compatible orchestrator over the local model fleet., _chunk_sse(), create_app(), _error(), main(), OpenAI-compatible orchestrator server: route -> shape -> relay. (+2 more)

### Community 7 - "Stub Test Backend"
Cohesion: 0.25
Nodes (4): BaseHTTPRequestHandler, main(), make_handler(), Minimal standalone OpenAI-compatible stub backend for smoke-testing the orchestr

### Community 8 - "GLM Streaming Synthesis"
Cohesion: 0.67
Nodes (3): glm greedy-only, synthesized SSE streaming, server.py, _synthesized_sse (server.py)

## Knowledge Gaps
- **15 isolated node(s):** `orchestrator`, `shaping.py`, `server.py`, `Model aliases table`, `Autoload (first-request cold start)` (+10 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Registry` connect `Backend Registry & Lifecycle` to `Router Test Suite`, `Config & Routing Core`, `HTTP Server & SSE`?**
  _High betweenness centrality (0.161) - this node is a cross-community bridge._
- **Why does `Cfg` connect `Config & Routing Core` to `Backend Registry & Lifecycle`, `HTTP Server & SSE`?**
  _High betweenness centrality (0.034) - this node is a cross-community bridge._
- **Why does `load()` connect `Config & Routing Core` to `HTTP Server & SSE`?**
  _High betweenness centrality (0.013) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `Cfg` (e.g. with `Registry` and `Route`) actually correct?**
  _`Cfg` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `Route` (e.g. with `Cfg` and `RoutingCfg`) actually correct?**
  _`Route` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `orchestrator`, `shaping.py`, `server.py` to the rest of the system?**
  _15 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Router Test Suite` be split into smaller, more focused modules?**
  _Cohesion score 0.08172043010752689 - nodes in this community are weakly interconnected._