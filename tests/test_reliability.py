"""Tests for backend self-recovery: health-probe debounce, crash-restart
backoff, the autoload circuit breaker, and defensive JSON parsing."""
from __future__ import annotations

import asyncio
import copy
import math
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from orchestrator import config
from orchestrator.backends import Registry
from orchestrator.server import _parse_json_bytes

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


@pytest.fixture(scope="module")
def cfg() -> config.Cfg:
    return config.load(CONFIG_PATH)


def _dead_proc(code: int = 1):
    return SimpleNamespace(poll=lambda: code, returncode=code, pid=999)


class _FailClient:
    async def get(self, *a, **kw):
        raise ConnectionError("down")


class _OkClient:
    async def get(self, *a, **kw):
        return SimpleNamespace(status_code=200)


# -- health probe debounce ----------------------------------------------------


def test_probe_single_failure_does_not_evict_resident(cfg):
    registry = Registry(cfg, client=_FailClient())
    registry.resident = {"ornith"}
    asyncio.run(registry._probe("ornith"))
    # fail_threshold default is 2 - one failure alone must not evict, or a
    # single busy-probe blip would flap routing.
    assert "ornith" in registry.resident
    assert registry.health["ornith"].consecutive_fail == 1


def test_probe_evicts_after_fail_threshold(cfg):
    registry = Registry(cfg, client=_FailClient())
    registry.resident = {"ornith"}
    for _ in range(cfg.health.fail_threshold):
        asyncio.run(registry._probe("ornith"))
    assert "ornith" not in registry.resident


def test_probe_success_clears_all_failure_bookkeeping(cfg):
    registry = Registry(cfg, client=_OkClient())
    h = registry.health["ornith"]
    h.consecutive_fail = 5
    h.restart_attempts = 3
    h.next_restart_at = 999999.0
    h.autoload_fail = 2
    h.autoload_open_until = 999999.0

    asyncio.run(registry._probe("ornith"))

    assert "ornith" in registry.resident
    assert h.consecutive_fail == 0
    assert h.restart_attempts == 0
    assert h.next_restart_at == 0.0
    assert h.autoload_fail == 0
    assert h.autoload_open_until == 0.0


# -- crash-restart watchdog ---------------------------------------------------


def test_supervise_relaunches_crashed_managed_backend(cfg, tmp_path):
    registry = Registry(cfg, client=None)
    registry.log_dir = tmp_path
    registry.procs["ornith"] = _dead_proc()
    launched = []
    registry.launch = lambda name, log_dir, profile=None: launched.append(name) or "ok"

    registry._supervise()

    assert launched == ["ornith"]
    assert registry.health["ornith"].restart_attempts == 1
    assert "ornith" not in registry.procs  # popped pre-relaunch; launch() re-populates in real code


def test_supervise_skips_still_alive_processes(cfg, tmp_path):
    registry = Registry(cfg, client=None)
    registry.log_dir = tmp_path
    registry.procs["ornith"] = SimpleNamespace(poll=lambda: None, returncode=None, pid=1)
    launched = []
    registry.launch = lambda name, log_dir, profile=None: launched.append(name) or "ok"

    registry._supervise()

    assert launched == []


def test_supervise_backoff_is_exponential_and_capped(cfg, tmp_path, monkeypatch):
    fake_now = [1000.0]
    monkeypatch.setattr("orchestrator.backends.time.monotonic", lambda: fake_now[0])

    registry = Registry(cfg, client=None)
    registry.log_dir = tmp_path
    launches = []
    registry.launch = lambda name, log_dir, profile=None: launches.append(name) or "ok"

    base = cfg.health.restart_backoff_s
    cap = cfg.health.restart_max_backoff_s
    h = registry.health["ornith"]

    # Crash #1
    registry.procs["ornith"] = _dead_proc()
    registry._supervise()
    assert launches == ["ornith"]
    assert h.restart_attempts == 1
    assert h.next_restart_at == pytest.approx(fake_now[0] + base)

    # Still inside the backoff window - a new crash must NOT trigger another
    # restart attempt yet.
    fake_now[0] += base / 2
    registry.procs["ornith"] = _dead_proc()
    registry._supervise()
    assert launches == ["ornith"]  # unchanged

    # Advance past the window - crash #2 should double the backoff.
    fake_now[0] += base
    registry.procs["ornith"] = _dead_proc()
    registry._supervise()
    assert launches == ["ornith", "ornith"]
    assert h.restart_attempts == 2
    assert h.next_restart_at == pytest.approx(fake_now[0] + base * 2)

    # Keep crashing past the window each time until backoff caps out.
    n_needed = math.ceil(math.log2(cap / base)) + 2
    for _ in range(n_needed):
        fake_now[0] = h.next_restart_at + 0.01
        registry.procs["ornith"] = _dead_proc()
        registry._supervise()
    assert h.next_restart_at - fake_now[0] == pytest.approx(cap, rel=0.01)


def test_supervise_skips_when_restart_disabled(cfg, tmp_path):
    local_cfg = copy.deepcopy(cfg)
    local_cfg.models["ornith"].restart = False

    registry = Registry(local_cfg, client=None)
    registry.log_dir = tmp_path
    launched = []
    registry.launch = lambda name, log_dir, profile=None: launched.append(name) or "ok"
    registry.procs["ornith"] = _dead_proc()
    registry.resident.add("ornith")

    registry._supervise()

    assert launched == []
    assert "ornith" not in registry.procs
    assert "ornith" not in registry.resident


def test_supervise_defers_when_conflict_occupies_memory(cfg, tmp_path):
    registry = Registry(cfg, client=None)
    registry.log_dir = tmp_path
    # A crashed ds4-full cannot restart while Ornith is resident. ds4-light is
    # intentionally compatible, so record the active full profile explicitly.
    registry.resident = {"ornith"}
    launched = []
    registry.launch = lambda name, log_dir, profile=None: launched.append(name) or "ok"
    registry.procs["ds4"] = _dead_proc()
    registry.active_profiles["ds4"] = "ds4-full"

    registry._supervise()

    assert launched == []
    assert "ds4" not in registry.procs


def test_supervise_noop_without_log_dir(cfg, tmp_path):
    registry = Registry(cfg, client=None)
    assert registry.log_dir is None
    registry.procs["ornith"] = _dead_proc()
    launched = []
    registry.launch = lambda name, log_dir, profile=None: launched.append(name) or "ok"

    registry._supervise()  # must not raise or attempt anything without a log_dir

    assert launched == []


# -- autoload circuit breaker -------------------------------------------------


def test_circuit_opens_after_max_failures(cfg, monkeypatch, tmp_path):
    registry = Registry(cfg, client=None)
    attempts = []

    async def fake_launch_for_load(name, log_dir, profile=None):
        attempts.append(name)
        return "launched"

    async def fake_wait_resident(name, timeout_s):
        return False  # never comes up

    monkeypatch.setattr(registry, "launch_for_load", fake_launch_for_load)
    monkeypatch.setattr(registry, "wait_resident", fake_wait_resident)

    max_fail = cfg.health.autoload_max_failures
    for _ in range(max_fail):
        assert asyncio.run(registry.ensure_resident("ds4", tmp_path)) is False

    assert len(attempts) == max_fail
    assert registry.health["ds4"].autoload_open_until > time.monotonic()

    # Circuit is now open - the next call must fast-fail WITHOUT a new launch.
    assert asyncio.run(registry.ensure_resident("ds4", tmp_path)) is False
    assert len(attempts) == max_fail


def test_circuit_fast_fails_without_launch_attempt_while_open(cfg, monkeypatch, tmp_path):
    registry = Registry(cfg, client=None)
    registry.health["ds4"].autoload_open_until = time.monotonic() + 30.0
    called = []

    async def fake_launch_for_load(name, log_dir, profile=None):
        called.append(name)
        return "launched"

    monkeypatch.setattr(registry, "launch_for_load", fake_launch_for_load)

    assert asyncio.run(registry.ensure_resident("ds4", tmp_path)) is False
    assert called == []


def test_circuit_closes_on_successful_load(cfg, monkeypatch, tmp_path):
    registry = Registry(cfg, client=None)
    registry.health["ds4"].autoload_fail = 2  # prior failures, circuit not yet open

    async def fake_launch_for_load(name, log_dir, profile=None):
        return "launched"

    async def fake_wait_resident(name, timeout_s):
        registry.resident.add(name)
        return True

    monkeypatch.setattr(registry, "launch_for_load", fake_launch_for_load)
    monkeypatch.setattr(registry, "wait_resident", fake_wait_resident)

    assert asyncio.run(registry.ensure_resident("ds4", tmp_path)) is True
    assert registry.health["ds4"].autoload_fail == 0
    assert registry.health["ds4"].autoload_open_until == 0.0


def test_probe_success_closes_open_circuit(cfg):
    # Any path that finds the backend healthy again (explicit /admin/load,
    # someone restarting it by hand) should close the circuit, not just the
    # autoload path specifically.
    registry = Registry(cfg, client=_OkClient())
    registry.health["ds4"].autoload_open_until = time.monotonic() + 30.0
    asyncio.run(registry._probe("ds4"))
    assert registry.health["ds4"].autoload_open_until == 0.0


# -- status() surfacing --------------------------------------------------


def test_status_hides_trouble_fields_when_healthy(cfg):
    registry = Registry(cfg, client=None)
    status = registry.status()
    for field in ("consecutive_probe_failures", "restart_attempts",
                  "restart_backoff_s", "autoload_circuit_open_s"):
        assert field not in status["ornith"]


def test_status_surfaces_trouble_fields_when_active(cfg):
    registry = Registry(cfg, client=None)
    h = registry.health["ds4"]
    h.consecutive_fail = 1
    h.restart_attempts = 2
    h.next_restart_at = time.monotonic() + 5.0
    h.autoload_open_until = time.monotonic() + 10.0

    status = registry.status()
    assert status["ds4"]["consecutive_probe_failures"] == 1
    assert status["ds4"]["restart_attempts"] == 2
    assert status["ds4"]["restart_backoff_s"] > 0
    assert status["ds4"]["autoload_circuit_open_s"] > 0


# -- defensive JSON parsing ---------------------------------------------------


def test_parse_json_bytes_valid():
    data, ok = _parse_json_bytes(b'{"a": 1}')
    assert ok is True
    assert data == {"a": 1}


def test_parse_json_bytes_empty_is_ok_none():
    data, ok = _parse_json_bytes(b"")
    assert ok is True
    assert data is None


def test_parse_json_bytes_malformed_is_not_ok():
    data, ok = _parse_json_bytes(b"<html>not json</html>")
    assert ok is False
    assert data is None
