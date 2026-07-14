"""Backend residency tracking and explicit-swap process management.

Residency is probed in the background so routing decisions read a cached
in-memory set and never block on a health check."""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx

from .config import Cfg

log = logging.getLogger("orchestrator.backends")


class Registry:
    def __init__(self, cfg: Cfg, client: httpx.AsyncClient):
        self.cfg = cfg
        self.client = client
        self.resident: set[str] = set()
        self.latency_ms: dict[str, float] = {}
        self.procs: dict[str, subprocess.Popen] = {}
        self.active: dict[str, int] = {}
        self._active_changed = asyncio.Condition()
        self._task: asyncio.Task | None = None
        self._launch_locks: dict[str, asyncio.Lock] = {}

    async def start(self):
        await self.probe_all()
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        if self._task:
            self._task.cancel()

    async def _loop(self):
        while True:
            await asyncio.sleep(self.cfg.health.interval_s)
            try:
                await self.probe_all()
            except Exception:
                log.exception("health probe pass failed")

    async def probe_all(self):
        await asyncio.gather(*(self._probe(name) for name in self.cfg.models))

    async def _probe(self, name: str):
        m = self.cfg.models[name]
        t0 = time.monotonic()
        try:
            r = await self.client.get(
                f"{m.upstream.rstrip('/')}/models",
                timeout=self.cfg.health.probe_timeout_s,
            )
            ok = r.status_code == 200
        except Exception:
            ok = False
        if ok:
            self.resident.add(name)
            self.latency_ms[name] = (time.monotonic() - t0) * 1000
        else:
            self.resident.discard(name)
            self.latency_ms.pop(name, None)

    async def wait_resident(self, name: str, timeout_s: float) -> bool:
        """Poll until `name` shows resident or timeout_s elapses. For callers
        (agent frameworks) that must not send their first request until the
        backend is actually up, since a mid-session cold-start 503 mid-agent-loop
        is worse than a slow synchronous wait before starting."""
        deadline = time.monotonic() + timeout_s
        while True:
            if name in self.resident:
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            await self._probe(name)
            if name in self.resident:
                return True
            await asyncio.sleep(min(self.cfg.health.probe_timeout_s, remaining))

    async def ensure_resident(self, name: str, log_dir: Path) -> bool:
        """First-request autoload: launch (idempotent) + block until ready,
        serialized per model so concurrent requests for the same cold model
        wait on one launch instead of racing duplicate subprocesses."""
        if name in self.resident:
            return True
        m = self.cfg.models[name]
        if not m.enabled or not m.autoload or not m.launch_cmd:
            return False
        lock = self._launch_locks.setdefault(name, asyncio.Lock())
        async with lock:
            if name in self.resident:
                return True
            try:
                await self.launch_for_load(name, log_dir)
            except ValueError:
                log.exception("autoload failed for %s", name)
                return False
            return await self.wait_resident(name, m.load_timeout_s)

    # -- explicit swap ------------------------------------------------------

    @asynccontextmanager
    async def active_request(self, name: str):
        async with self._active_changed:
            self.active[name] = self.active.get(name, 0) + 1
            self._active_changed.notify_all()
        try:
            yield
        finally:
            async with self._active_changed:
                next_count = self.active.get(name, 0) - 1
                if next_count > 0:
                    self.active[name] = next_count
                else:
                    self.active.pop(name, None)
                self._active_changed.notify_all()

    async def wait_idle(self, name: str, timeout_s: float | None = None) -> bool:
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        async with self._active_changed:
            while self.active.get(name, 0) > 0:
                if deadline is None:
                    await self._active_changed.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                try:
                    await asyncio.wait_for(self._active_changed.wait(), remaining)
                except asyncio.TimeoutError:
                    return False
            return True

    async def _unload_before_load(self, name: str):
        m = self.cfg.models[name]
        for other in m.unload_before_load:
            if other == name:
                continue
            if other not in self.cfg.models:
                raise ValueError(f"{name}.unload_before_load references unknown model {other!r}")

            # Refresh before deciding. A backend may have gone down since the
            # last background probe.
            await self._probe(other)
            if other not in self.resident:
                continue

            idle = await self.wait_idle(other, 60.0)
            if not idle:
                raise ValueError(
                    f"{name} requires unloading {other} first, but {other} still has "
                    f"{self.active.get(other, 0)} active request(s). Retry after it is idle."
                )

            proc = self.procs.get(other)
            if proc and proc.poll() is None:
                self.terminate(other)
                await self._probe(other)
                continue

            raise ValueError(
                f"{name} requires unloading {other} first, but {other} is resident "
                "and was not launched by this orchestrator process. Stop it manually "
                "before loading this model."
            )

    def launch(self, name: str, log_dir: Path) -> str:
        m = self.cfg.models[name]
        if not m.launch_cmd:
            raise ValueError(f"{name} has no launch_cmd configured")
        proc = self.procs.get(name)
        if proc and proc.poll() is None:
            return f"{name} already launched (pid {proc.pid})"
        log_dir.mkdir(parents=True, exist_ok=True)
        logfile = open(log_dir / f"{name}.log", "ab")
        proc = subprocess.Popen(
            m.launch_cmd, shell=True, start_new_session=True,
            stdout=logfile, stderr=subprocess.STDOUT,
        )
        self.procs[name] = proc
        log.info("launched %s pid=%d", name, proc.pid)
        return f"launched {name} (pid {proc.pid}); resident once its /v1/models responds"

    async def launch_for_load(self, name: str, log_dir: Path) -> str:
        await self._unload_before_load(name)
        return self.launch(name, log_dir)

    async def unload_when_idle(self, name: str, timeout_s: float = 0.0) -> str:
        idle = await self.wait_idle(name, timeout_s)
        if not idle:
            raise ValueError(
                f"{name} still has {self.active.get(name, 0)} active request(s); "
                "retry later or use force=true"
            )
        return self.terminate(name)

    def terminate(self, name: str) -> str:
        proc = self.procs.get(name)
        if not proc or proc.poll() is not None:
            raise ValueError(
                f"{name} was not launched by the orchestrator "
                "(externally-started backends must be stopped externally)"
            )
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        self.procs.pop(name, None)
        self.resident.discard(name)
        return f"sent SIGTERM to {name} process group (pid {proc.pid})"

    def status(self) -> dict:
        out = {}
        for name, m in self.cfg.models.items():
            proc = self.procs.get(name)
            out[name] = {
                "resident": name in self.resident,
                "upstream": m.upstream,
                "probe_latency_ms": round(self.latency_ms[name], 1) if name in self.latency_ms else None,
                "managed_pid": proc.pid if proc and proc.poll() is None else None,
                "active_requests": self.active.get(name, 0),
                "tags": m.tags,
            }
        return out
