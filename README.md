# orchestrator

An OpenAI-compatible HTTP proxy that routes chat/completion requests across a
fleet of local model backends (shipinabottle GLM, llama.cpp servers). It picks
a backend per-request with cheap heuristics, adapts the request for whatever
quirks that backend has, and relays the response — streaming or not.

## Quickstart

```bash
cd /home/quinna/tools/orchestrator
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/orchestrator                      # or: .venv/bin/python -m orchestrator.server
```

Flags: `--config PATH` (default `config.yaml` next to the package),
`--host`, `--port` (default from `config.yaml`, currently `0.0.0.0:8080`).

Point an `openai` SDK client at it. By default the orchestrator doesn't
check the API key (any placeholder works); set `api_key:` in `config.yaml`
to require a real bearer token (see [Auth and CORS](#auth-and-cors)).

```python
from openai import OpenAI

client = OpenAI(base_url="http://HOST:8080/v1", api_key="unused")
client.chat.completions.create(model="auto", messages=[...])
```

## API surface

OpenAI-compatible endpoints under `/v1`:

| Endpoint | Notes |
|---|---|
| `POST /v1/chat/completions` | Main path. Routing, autoload, fallback, streaming, vision-subagent, model-name rewrite all apply. |
| `POST /v1/completions` | Legacy text completions. Routing minus the tools/vision dimensions. |
| `POST /v1/embeddings` | Routed to any model with `embeddings: true` (prefers a warm one, autoloads if needed). Pin a specific one via `model`; a non-embeddings pin returns a clear `400`. |
| `GET /v1/models` | Lists backends + aliases, each with `resident`, `tags`, `embeddings`, `vision` flags. |
| `GET /v1/models/{id}` | Single-model detail; `404` for unknown ids. |

Operational endpoints (also behind auth when enabled, except `/health`):

| Endpoint | Notes |
|---|---|
| `GET /health` | Liveness — always open, even with auth on. Returns resident set. |
| `GET /admin/status` | Per-model resident flag, active/default launch profile, probe latency, managed PID, and in-flight request count. |
| `GET /admin/stats` | Usage accounting: per-model request/error/token/latency totals, plus a ring buffer of recent routing decisions (each with `reason`, `text_chars`, `preferred`, token usage). See [Observability](#observability-and-tuning). |
| `POST /admin/load` / `POST /admin/unload` | Explicit residency control (`wait_s`, `wait_idle_s`, `force`). |

## Auth and CORS

Two `config.yaml` top-level keys:

- **`api_key`** (default `null` = off). When set, every `/v1/*` and
  `/admin/*` request must send `Authorization: Bearer <key>`; a missing or
  wrong key gets `401`. `/health` stays open so liveness probes and load
  balancers still work. Fine to leave off on a trusted LAN; set one if the
  `0.0.0.0` bind is reachable from anywhere you don't fully control.
- **`cors`** (default `true`). Allows browser-based dashboards to call the
  API cross-origin, and exposes the `x-orchestrator-*` response headers to
  browser JS so a dashboard can read which model actually served a request.
  `OPTIONS` preflight requests bypass the auth check (they carry no auth
  header by design).

## Observability and tuning

`GET /admin/stats` returns two things:

- **`models`** — per-backend totals: `requests`, `errors`, `prompt_tokens`,
  `completion_tokens`, `avg_latency_ms`, `last_used`. A quick read on which
  backends are carrying load and which are erroring.
- **`recent`** — the last ~200 routing decisions, each with the `reason`
  tag, the request's `text_chars`, the `preferred` model (when a fallback
  happened), whether it `autoloaded`, and token `usage`. This is the data
  for tuning the escalation thresholds: e.g. to decide whether
  `precise_min_chars` is set right, look at what `text_chars` the
  `long-prompt`-reason and `default`-reason requests actually cluster at,
  rather than guessing. It's in-memory only (resets on restart) — for
  durable history, scrape it on an interval.

## Routing

Routing decisions happen in `orchestrator/router.py`, in this order:

1. **Explicit real model name** (`glm`, `ornith`, `ds4`, `gemma`) — pinned,
   no heuristics, no fallback. If that backend isn't resident, the request
   fails with `503` instead of silently landing elsewhere.
2. **Alias** (see table below) that isn't `auto` — resolved to its target
   model, still subject to normal (non-forced) fallback.
3. Otherwise (name is `auto`, or unrecognized, e.g. `gpt-4o`) — heuristic
   routing on message content:

| Condition | Route | Reason tag |
|---|---|---|
| A content part with a **non-empty** image url/data | `gemma` (or `default` if vision unconfigured) | `vision` / `vision-unconfigured` |
| `tools`/`functions` present or `response_format.type == "json_schema"`, and complex (≥3 tools, ≥4000 chars of text, or an agent loop already in progress — a message with `role: "tool"`) | `ds4` | `tools-complex` |
| `tools`/`functions` present, not complex | `ornith` | `tools-simple` |
| Text matches a "precise" pattern (proof/theorem/derive, "exact"/"precise"/"verbatim"/"double-check", LaTeX math, math symbols) | `glm` | `precise` |
| Nothing else matched | `ornith` (config `routing.default`) | `default` |

`/v1/completions` (legacy, text-only) only checks explicit pin, alias, and
the precise-pattern rule; everything else falls to `default`.

Thresholds (`tools_complex_min_tools`, `tools_complex_min_chars`) and the
precise-pattern regex list live in `config.yaml` under `routing:` and are
tunable without touching code.

**Empty image slots don't count as images.** Many chat UIs always attach an
`image_url` content part structurally — an empty attachment slot — whether
or not the user actually picked an image. `router.has_real_image()` checks
for an actual non-empty url/data before treating a part as a real image, so
those requests route normally instead of always landing on `gemma`.
`shaping.py` goes further and strips any such empty image parts from the
outgoing request entirely (not just excludes them from routing) — llama.cpp
backends without `--mmproj` (`ornith`, `ds4`) 500 outright if the request
structurally contains an image part, regardless of whether it's empty, so
routing correctly around it isn't enough; the fake part has to be removed
before the request is forwarded anywhere. Real image parts pass through
untouched either way.

## Explicitly choosing a model

Set `model` to a real backend name (`glm`, `ornith`, `ds4`, `gemma`) or an
alias (table below) in any request — this is a forced pin: no heuristics
run, and if that backend isn't resident it autoloads (see below) rather
than silently landing on a different model. This already works with any
OpenAI-compatible client/dashboard that lets you type or select a model
name — nothing extra needed on the client side.

## Model aliases

Client-facing `model` values that map to a backend or to `auto`:

| Alias | Target |
|---|---|
| `auto`, `default`, `orchestrator` | `auto` (heuristic routing) |

Direct backend pins are still available as real model names: `ornith`,
`ds4`, `gemma`, and `glm`. A real backend pin is forced: it goes only to
that backend and fails if that backend is unavailable.

Any other unrecognized name (including removed mode aliases like `fast`,
`precise`, `agent`, `vision`, or `gpt-*`) also auto-routes. This keeps
drop-in OpenAI clients working without listing every mode-ish alias in
`/v1/models`.

## Resident set, autoload, and explicit swaps

The orchestrator never assumes a backend is up. A background probe
(`health.interval_s`, default 15s) hits each backend's `/v1/models` and
maintains an in-memory `resident` set; routing reads that set, it never
blocks on a live health check.

- **Forced routes** (explicit pin) require the pinned backend to be
  resident (or autoload it — see below). If it's still not resident after
  that, the request fails with `503 model_not_resident` — no silent
  fallback for a pin.
- **Non-forced routes** (alias or heuristic) walk the target model's
  `fallback` chain in `config.yaml` (e.g. `ds4` → `ornith`) and use the
  first *enabled* backend in that chain that's resident. If a fallback
  backend actually served the request, the response carries an
  `x-orchestrator-preferred` header naming the originally-preferred model.

**Autoload (first-request cold start):** if nothing in the chain is
resident, the orchestrator doesn't immediately 503 — it tries each
candidate in order and, for the first one with `autoload: true` (default)
and a configured `launch_cmd`, launches it and blocks *that one request*
until `/v1/models` responds (bounded by that model's `load_timeout_s`,
default 180s) before proceeding. Concurrent requests for the same cold
model share one launch (serialized per-model, no duplicate subprocesses).
The response carries `x-orchestrator-autoloaded: true` when this happened,
so you can tell a slow first response apart from a genuinely slow model.
Set `autoload: false` on a model to opt it out (e.g. `glm` — see below);
it then only ever comes up via explicit `/admin/load`.

Endpoints for managing residency directly:

- `GET /admin/status` — per-model resident flag, upstream URL, last probe
  latency, managed PID (if the orchestrator launched it), active request
  count, tags.
- `POST /admin/load {"model": "ornith"}` — runs that model's `launch_cmd`
  as a detached subprocess (own process group) and logs to
  `orchestrator/logs/<model>.log`. Fails if no `launch_cmd` is configured
  or if already launched. The model becomes "resident"
  once its `/v1/models` starts responding — not immediately after launch.
  Add `"wait_s": N` to block the call itself until ready (or `504` on
  timeout) instead of returning immediately.
- `POST /admin/unload {"model": "ornith", "wait_idle_s": 300}` — waits
  until that backend has no active in-flight requests, then sends `SIGTERM`
  to the process group. With no `wait_idle_s`, it unloads only if already
  idle. Only works for backends the orchestrator itself launched;
  externally-started backends must be stopped externally.
- `POST /admin/unload {"model": "ornith", "force": true}` — immediate
  `SIGTERM` without waiting for active requests to finish.

### Named launch profiles

A logical model can have several named runtime profiles in `config.yaml` under
`launch_profiles:`. Routing and client requests still use the logical model
name; profiles only choose the process command and memory-conflict policy.
This keeps a model's OpenAI/API behavior stable while making operational
tradeoffs explicit.

DS4 includes two profiles:

- `ds4-light` is the default autoload profile. It uses `--ssd-streaming`, has
  lower memory use, and may coexist with Ornith.
- `ds4-full` omits SSD streaming for fully resident weights and higher decode
  throughput. It conflicts with Ornith, GLM, and Gemma, so the registry swaps
  those managed processes out first.

Profiles are also standard OpenAI-compatible model IDs. Any client can select
one with its normal `model` field — for example Hermes can set
`model.default: ds4-full` while keeping `base_url: http://127.0.0.1:8080/v1`.
An explicit profile pin loads or switches to that managed profile when idle;
the completion and `x-orchestrator-profile` response header identify it.

Load a specific profile before an agent session:

```sh
curl -X POST http://127.0.0.1:8080/admin/load \
  -H 'Content-Type: application/json' \
  -d '{"model":"ds4","profile":"ds4-full","wait_s":60}'
```

The shorter form `{"profile":"ds4-full"}` is also accepted. A model must be
unloaded before switching its profile because both DS4 profiles use the same
upstream port. The registry treats profile conflicts symmetrically, preventing
an order-dependent attempt to load Ornith beside an already-resident
`ds4-full`.

### GLM is currently disabled

`glm.enabled: false` in `config.yaml`, per Quinn: sub-1-token/s decode
makes it unusable in the automatic paths for now. This means `resolve()`
skips it in any non-forced fallback chain (`routing.precise` requests fall
through to `ornith` instead, with `x-orchestrator-preferred: glm` still
shown so you can see what the "ideal" routing would have been), and
`autoload: false` means it's never auto-launched. It's still directly
reachable — `model: "glm"` explicitly pins to it and `/admin/load` will
still launch it — since those are deliberate choices, not automatic
routing. Flip `enabled`/`autoload` back to `true` once decode speed is
fixed; nothing else needs to change.

## Which model actually answered

The response body's `model` field is rewritten to the orchestrator's
canonical name (`ornith`, `gemma`, `glm`, `ds4`) — for **every** response
shape: plain JSON, native SSE streaming (rewritten chunk-by-chunk), and
synthesized SSE (glm). This matters because backends echo their own idea of
"model": llama.cpp backends return the literal `.gguf` file path, not a
clean name. If your client/dashboard reads `response.model` (the standard
OpenAI field — most do, since it's what displays in a chat UI's "model
used" indicator), it now shows a correct, readable name automatically, no
extra parsing needed.

Every relayed response (streaming or not) also carries these headers:

- `x-orchestrator-model` — the backend that actually served the request
  (same value as the rewritten body `model` field).
- `x-orchestrator-reason` — why it was routed there (`default`,
  `vision`, `vision-unconfigured`, `tools-simple`, `tools-complex`,
  `precise`, `client-pinned`, `alias:<name>`).
- `x-orchestrator-preferred` — only present when the serving backend
  differs from the originally preferred one (i.e. a fallback happened).
- `x-orchestrator-autoloaded` — only present (`"true"`) when this specific
  request had to launch the backend itself before it could be served.
- `x-orchestrator-retried-after` — only present when a candidate backend
  was resident but refused the connection mid-request, and the orchestrator
  transparently retried the next candidate in the chain. Names the
  backend(s) that failed. See [mid-request retry](#mid-request-retry).

### Mid-request retry

A backend can pass its health probe and then die before (or during) the
actual request. When a resident candidate refuses the connection, the
orchestrator doesn't surface a `502` immediately — it evicts that backend
from the resident set and tries the next candidate in the same chain the
router would have used for fallback (preferred model + its enabled
fallbacks). Only if *every* candidate refuses does the request fail. The
successful response carries `x-orchestrator-retried-after` naming what
failed, so a transparently-recovered request is still visible as one that
had trouble. A timeout (as opposed to a connection refusal) is *not*
retried — a backend that accepted the request may be mid-generation, and
resending would double the work.

## Per-model parameters and persistent context

Four fields on each model in `config.yaml`, all applied by `shaping.shape()`
before the request leaves the orchestrator:

- **`sampling_defaults`** — applied only if the client's request didn't
  already set that key. Works for *any* field, not a fixed list (e.g.
  `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `stop`,
  `seed` all work identically). Use this for a reasonable default that a
  caller should still be free to override.
- **`forced_params`** — same shape, but always wins, even overriding what
  the client explicitly sent. Use this to lock in behavior rather than fill
  a gap — e.g. `{temperature: 0}` on a model used for deterministic
  tool-calling. Applied last, after everything else (including
  `greedy_only`'s stripping), so it's the final word.
- **`system_prompt`** — this is "context set beforehand": a string injected
  as a single leading system message on *every* request to that model,
  server-side, so the client never has to send it. It's coalesced with any
  system/developer messages already in the conversation (or from Gemma's
  media subprompt, or Ornith's post-Gemma observation wrapper) into one
  message — most chat templates require exactly one leading system message,
  so nothing here should be written assuming it'll be sole content. `null`/
  omitted means no injected prompt, just whatever the client sent.
- **`force_max_tokens`** — older, narrower special case of the same idea as
  `sampling_defaults` (default-if-unset, specifically for `max_tokens`) kept
  around for shipinabottle's tiny upstream default. `sampling_defaults:
  {max_tokens: N}` would do the same thing generically.

Example — pin ds4 to deterministic sampling and remind it who it's serving:

```yaml
ds4:
  forced_params:
    temperature: 0
  system_prompt: >
    You are the reasoning/tool-calling escalation target for this
    orchestrator. Requests reach you because Ornith or the router judged
    them long, complex, or explicitly high-effort — engage accordingly.
```

All four are per-model, live in `config.yaml`, and need no code changes —
but `config.yaml` is only read once, at orchestrator process startup
(`main()` loads it and every request handler closes over that same `Cfg`
object for the process's lifetime). **Editing `config.yaml` requires
restarting the orchestrator process** to take effect. The model backends
themselves (ornith, gemma, etc.) do *not* need restarting — these fields
are applied per-request by the orchestrator, not baked into how a backend
was launched, so a warm backend keeps serving fine across an orchestrator
restart.

## FILL_ME placeholders

One left: **`ds4.launch_cmd`**'s `gguf/FILL_ME_QUANT.gguf` — the DeepSeek
V4 weights (`models/ds4/download_model.sh`) haven't finished downloading.
Everything else about `ds4` is already correct: it's not served by vllm or
llama.cpp. `models/ds4` is antirez's own bespoke Metal/CUDA inference
engine for DeepSeek V4 Flash (same category as shipinabottle —
purpose-built, not a generic runner), and it ships its own
OpenAI/Anthropic/Responses-compatible server binary, `ds4-server`, already
built. Once a quant finishes downloading, drop its filename in.
`q2-imatrix` (~81 GB) is the one that actually fits alongside `glm` +
`ornith` on this box's 121 GB unified memory; the configured
`--ssd-streaming` flag (native popularity-cached routed-expert streaming +
disk KV checkpoints — ds4's own answer to the same "can't fit everything
in RAM" problem shipinabottle solves for GLM) can be dropped if a smaller
quant ends up fully resident instead.

### Why gemma moved off vllm

vllm's per-launch cost (weight materialization into GPU tensors, CUDA graph
capture, engine warmup) is real and, per Quinn, slow in practice here. This
checkout of `~/llama.cpp` already had first-class Gemma4 support —
`Gemma4Model` and a registered `Gemma4VisionAudioModel` (the `--mmproj`
extractor) in `conversion/gemma.py`, `llama-server` and
`llama-gemma3-cli` already built — so `gemma` now follows the exact same
pattern as `ornith`: a quantized GGUF, mmap-loaded, served by
`llama-server`. Converted 2026-07-13 from the HF safetensors checkpoint
(`google/gemma-4-31B-it`, already fully downloaded — an earlier note in
this file wrongly said the download was incomplete; that was a stale
orphaned `.incomplete` temp file from an interrupted/duplicate download
attempt, unrelated to the live snapshot) using the vllm venv's existing
torch/transformers/sentencepiece/protobuf, with only `gguf-py` added via
`PYTHONPATH` (no new installs). One gotcha worth knowing if you ever
reconvert: `--mmproj` exports *only* the vision projector, not both files
in one run — it takes two separate `convert_hf_to_gguf.py` invocations.
Final artifacts: `gemma4-Q4_K_M.gguf` (17.8 GiB, 4.87 BPW) and
`mmproj-gemma4-f16.gguf` (1.2 GB), both under `models/gemma/`.

**Verified working end-to-end with the real model**, not just routing
logic: loaded via the orchestrator's own `/admin/load` (blocking on
`wait_s`, ~75s to come up), then a real image sent through the public
`/v1/chat/completions` with `model: "auto"` correctly routed to gemma and
got a correct answer back from actual vision inference. Image input is
unaffected by which engine serves gemma either way — `shaping.py` never
touches the `messages` array, so `image_url`/base64 content parts reach
whichever backend is chosen byte-for-byte regardless of what serves it
upstream.

Resolved already: `ornith` uses the real filename
`ornith-1.0-35b-Q4_K_M.gguf`. `glm.launch_cmd` is Quinn's standard
shipinabottle invocation on `127.0.0.1:8022`, matching `glm.upstream`; if
you change one, change the other.

## Why the orchestrator doesn't do model loading itself

It was tempting to make the orchestrator itself manage weight residency
the way shipinabottle does for GLM — but that would mean reimplementing a
bespoke, architecture-specific loading/caching engine for every backend,
which is exactly what shipinabottle (dense-weight quantized caching,
hot-expert preload) and `ds4-server` (SSD streaming, popularity expert
cache, disk KV checkpoints) already are, each purpose-built for one model's
memory-access pattern over months of tuning. Redoing that generically at
the proxy layer would be strictly worse than what's already there.

Instead the orchestrator's job is scoped to what a thin layer can actually
do well:

- **Never trigger a load per request.** The resident-set-with-explicit-swap
  design means a routed request either hits an already-warm backend or gets
  a fast `503`/fallback — it never blocks a request on a cold start.
- **Pay the load cost once per session, not once per prompt**, by keeping
  backends as long-lived subprocesses (`/admin/load` / `/admin/unload`)
  instead of spawning-per-request.
- **Let each engine's own fast path do the work.** llama.cpp backends
  (`ornith`, `gemma`) mmap their GGUF/safetensors, so on this box's NVMe
  (~4-6 GB/s) a cold swap-in is bounded by how much of the model is actually
  touched during warmup, not the full file size. `glm` and `ds4` lean on
  their own custom streaming engines for the same reason.

If swap latency between `ornith` and `ds4` ever becomes the bottleneck, the
next real lever is a small resident-process LRU (keep N ≥ 2 warm instead of
always fully unloading one to load another) — not orchestrator-managed
weight caching.

## Using it for agents

Point any OpenAI-compatible agent framework at
`base_url="http://HOST:8080/v1"`. Two things matter for an agent session
specifically, both different from a one-off chat request:

1. **Pin the model, don't rely on `auto`.** Per-message heuristic routing
   is right for independent requests, but an agent loop is one continuous
   session — you don't want message 3 landing on a different backend than
   message 1 because a tool definition got added or the prompt crossed a
   length threshold. Use `model: "ds4"` (or the `agent` alias) explicitly
   for the whole session. `ornith` is also a reasonable pin for
   lighter/faster agent work per Ornith's own agentic-coding benchmarks.
2. **Guarantee residency before the loop starts, not on the first
   request.** A pinned-but-not-resident backend 503s immediately (by
   design — pins never silently fall back), which is the right failure
   mode for a mid-loop request but the wrong one to discover at the start
   of a session. Call `/admin/load` with `wait_s` first:

```python
import httpx
httpx.post("http://HOST:8080/admin/load", json={"model": "ds4", "wait_s": 180})
# blocks (up to 180s) until ds4's /v1/models responds, or returns 504

from openai import OpenAI
client = OpenAI(base_url="http://HOST:8080/v1", api_key="unused")
# now safe to run the whole agent loop pinned to ds4
client.chat.completions.create(model="ds4", messages=[...], tools=[...])
```

`wait_s` also works if the backend is already resident (returns
immediately) or was started manually outside the orchestrator — it just
polls `/v1/models`, it doesn't require the orchestrator to have launched
the process.

## Should routing use a model instead of regex?

No, not by default — the whole point of heuristic routing is that it costs
microseconds, and every model on this box decodes at seconds-per-token, so
even a "fast" LLM router would tax every single request (including the
`ornith` fast-path ones it's least appropriate for) far more than the
regex/threshold checks it'd replace. It would also need its own memory
budget on a box that's already resident-constrained. If the regex rules
prove insufficient later, the right next step is a small *non-generative*
classifier (a few KB, sub-millisecond, CPU-only — e.g. a tiny sklearn/
fastText model over cheap text features) rather than a decoder LLM, or
escalating only genuinely ambiguous requests to a judgment call from
whichever fast model is already resident (`ornith`) instead of adding a new
resident model just for routing.

## glm: greedy-only, synthesized streaming

shipinabottle (the `glm` backend) rejects `stream: true` upstream (400) and
ignores sampling parameters — it's greedy decoding only. `shaping.py`
strips `temperature`/`top_p`/`presence_penalty`/`frequency_penalty`/
`logit_bias`/`seed` from requests routed to `glm`, and forces
`max_tokens=512` if the client didn't set one (upstream's own default of
16 truncates almost everything).

If a client asks `glm` to stream, the orchestrator makes a normal
non-streaming upstream call and re-emits the full completion as OpenAI-style
SSE chunks (`chat.completion.chunk` objects, then `data: [DONE]`) — see
`_synthesized_sse` in `server.py`. From the client's point of view it still
looks like a streaming response, just delivered in one burst once the
(potentially very slow — up to `glm.timeout_s`, 7200s) upstream call
returns.
