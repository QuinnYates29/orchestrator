from __future__ import annotations

from pathlib import Path

from orchestrator import config
from orchestrator.stats import Stats

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def test_stats_accumulates_per_model():
    s = Stats()
    s.record(endpoint="/chat/completions", model="ornith", reason="default",
             status=200, latency_ms=100.0, usage={"prompt_tokens": 10, "completion_tokens": 5})
    s.record(endpoint="/chat/completions", model="ornith", reason="default",
             status=200, latency_ms=300.0, usage={"prompt_tokens": 4, "completion_tokens": 6})
    snap = s.snapshot()
    m = snap["models"]["ornith"]
    assert m["requests"] == 2
    assert m["errors"] == 0
    assert m["prompt_tokens"] == 14
    assert m["completion_tokens"] == 11
    assert m["avg_latency_ms"] == 200.0


def test_stats_counts_errors():
    s = Stats()
    s.record(endpoint="/chat/completions", model="ds4", reason="precise",
             status=504, latency_ms=1000.0)
    assert s.snapshot()["models"]["ds4"]["errors"] == 1


def test_stats_recent_ring_records_reason_and_text_chars():
    s = Stats(recent_max=3)
    for i in range(5):
        s.record(endpoint="/chat/completions", model="ornith", reason="default",
                 status=200, latency_ms=1.0, text_chars=i)
    recent = s.snapshot()["recent"]
    assert len(recent) == 3          # ring capped
    assert recent[-1]["text_chars"] == 4
    assert recent[0]["reason"] == "default"


def test_stats_recent_flags_fallback_and_autoload():
    s = Stats()
    s.record(endpoint="/chat/completions", model="ornith", reason="tools-complex",
             status=200, latency_ms=1.0, preferred="ds4", autoloaded=True)
    entry = s.snapshot()["recent"][-1]
    assert entry["preferred"] == "ds4"
    assert entry["autoloaded"] is True


# -- new config fields -------------------------------------------------------


def test_config_defaults_for_new_fields():
    cfg = config.load(CONFIG_PATH)
    assert cfg.cors is True
    assert cfg.api_key is None
    # No deployed backend serves embeddings: llama-server's --embeddings
    # restricts the process to embeddings only, so turning it on for ornith
    # would take down the default chat model. ornith sets the field
    # explicitly, gemma leaves it at the default.
    assert cfg.models["ornith"].embeddings is False
    assert cfg.models["gemma"].embeddings is False
