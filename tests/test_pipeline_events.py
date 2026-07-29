from __future__ import annotations

import json

from pipeline.events import EventLog, NullEventLog


def test_emit_appends_jsonl_with_ts_and_run_id(tmp_path):
    log = EventLog(tmp_path / "events.jsonl", run_id="run-1")
    log.emit("plan", chunks=3)
    log.emit("chunk_finished", chunk="agent-1", status="completed")

    events = EventLog.read(log.path)
    assert [e["kind"] for e in events] == ["plan", "chunk_finished"]
    assert all(e["run_id"] == "run-1" for e in events)
    assert all(isinstance(e["ts"], float) for e in events)
    assert events[0]["chunks"] == 3


def test_for_run_places_log_under_run_id(tmp_path):
    log = EventLog.for_run(tmp_path, "20260728-120000-abc123")
    log.emit("run_start")
    assert log.path == tmp_path / "20260728-120000-abc123" / "events.jsonl"
    assert log.path.exists()


def test_unserializable_payload_does_not_raise(tmp_path):
    log = EventLog(tmp_path / "events.jsonl")
    log.emit("odd", value=object())          # default=str handles it
    assert len(EventLog.read(log.path)) == 1


def test_torn_final_line_is_skipped_not_fatal(tmp_path):
    """A run killed mid-write leaves a partial line; the log must stay readable."""
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps({"kind": "a", "ts": 1.0}) + "\n" + '{"kind": "b", "ts')
    events = EventLog.read(path)
    assert [e["kind"] for e in events] == ["a"]


def test_read_missing_file_is_empty(tmp_path):
    assert EventLog.read(tmp_path / "absent.jsonl") == []


def test_usage_totals_per_model(tmp_path):
    log = EventLog(tmp_path / "events.jsonl", run_id="r")
    log.emit_usage("worker", "ornith", {"prompt_tokens": 100, "completion_tokens": 20})
    log.emit_usage("worker", "ornith", {"prompt_tokens": 50, "completion_tokens": 5})
    log.emit_usage("planner", "ds4-full", {"prompt_tokens": 10, "completion_tokens": 400})

    totals = EventLog.token_totals(log.path)
    assert totals["ornith"] == {"prompt_tokens": 150, "completion_tokens": 25,
                                "calls": 2, "unreported_calls": 0}
    assert totals["ds4-full"]["completion_tokens"] == 400


def test_missing_usage_is_recorded_as_unreported_not_zero_cost(tmp_path):
    """A backend that omits usage must not read as a free call."""
    log = EventLog(tmp_path / "events.jsonl")
    log.emit_usage("worker", "ornith", None)
    totals = EventLog.token_totals(log.path)
    assert totals["ornith"]["calls"] == 1
    assert totals["ornith"]["unreported_calls"] == 1


def test_null_log_writes_nothing(tmp_path):
    log = NullEventLog()
    log.emit("anything", a=1)
    log.emit_usage("worker", "ornith", {"prompt_tokens": 1})
    assert EventLog.read(tmp_path / "events.jsonl") == []
