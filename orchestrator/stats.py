"""In-memory usage accounting and a recent-decisions ring buffer.

Two consumers: /admin/stats for at-a-glance per-model health (tokens,
latency, errors), and the `recent` list for tuning routing thresholds —
each entry carries the routing reason and the request's text length, so
questions like "what precise_min_chars separates real escalations from
noise" can be answered from observed traffic instead of guesses.
"""
from __future__ import annotations

import time
from collections import deque


class Stats:
    def __init__(self, recent_max: int = 200):
        self.per_model: dict[str, dict] = {}
        self.recent: deque = deque(maxlen=recent_max)
        self.started_at = time.time()

    def record(self, *, endpoint: str, model: str, reason: str, status: int,
               latency_ms: float, preferred: str | None = None,
               autoloaded: bool = False, usage: dict | None = None,
               text_chars: int | None = None) -> None:
        s = self.per_model.setdefault(model, {
            "requests": 0, "errors": 0,
            "prompt_tokens": 0, "completion_tokens": 0,
            "total_latency_ms": 0.0, "last_used": None,
        })
        s["requests"] += 1
        if status >= 400:
            s["errors"] += 1
        s["total_latency_ms"] += latency_ms
        s["last_used"] = time.time()
        if usage:
            s["prompt_tokens"] += usage.get("prompt_tokens") or 0
            s["completion_tokens"] += usage.get("completion_tokens") or 0

        entry = {
            "ts": round(time.time(), 3),
            "endpoint": endpoint,
            "model": model,
            "reason": reason,
            "status": status,
            "latency_ms": round(latency_ms, 1),
        }
        if preferred and preferred != model:
            entry["preferred"] = preferred
        if autoloaded:
            entry["autoloaded"] = True
        if text_chars is not None:
            entry["text_chars"] = text_chars
        if usage:
            entry["usage"] = {k: usage[k] for k in ("prompt_tokens", "completion_tokens")
                              if usage.get(k) is not None}
        self.recent.append(entry)

    def snapshot(self) -> dict:
        models = {}
        for name, s in self.per_model.items():
            avg = s["total_latency_ms"] / s["requests"] if s["requests"] else 0.0
            models[name] = {k: v for k, v in s.items() if k != "total_latency_ms"}
            models[name]["avg_latency_ms"] = round(avg, 1)
        return {
            "uptime_s": round(time.time() - self.started_at, 1),
            "models": models,
            "recent": list(self.recent),
        }
