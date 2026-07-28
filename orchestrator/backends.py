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
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .config import Cfg

log = logging.getLogger("orchestrator.backends")


@dataclass
class _Health:
    """Per-model runtime health used for debouncing, crash-restart backoff, and
    the autoload circuit breaker. Kept separate from static ModelCfg."""
    consecutive_fail: int = 0        # failed probes in a row (debounce down-marking)
    restart_attempts: int = 0        # crash-restarts in a row (exponential backoff)
    next_restart_at: float = 0.0     # monotonic; watchdog won't relaunch before this
    autoload_fail: int = 0           # failed autoloads in a row
    autoload_open_until: float = 0.0  # monotonic; circuit open (fast-fail) until this


class Registry:
    def __init__(self, cfg: Cfg, client: httpx.AsyncClient):
        self.cfg = cfg
        self.client = client
        self.resident: set[str] = set()
        self.latency_ms: dict[str, float] = {}
        self.procs: dict[str, subprocess.Popen] = {}
        self.active_profiles: dict[str, str] = {}
        self.active: dict[str, int] = {}
        self.health: dict[str, _Health] = {name: _Health() for name in cfg.models}
        self.log_dir: Path | None = None
        self._active_changed = asyncio.Condition()
        self._task: asyncio.Task | None = None
        self._launch_locks: dict[str, asyncio.Lock] = {}

    async def start(self, log_dir: Path | None = None):
        self.log_dir = log_dir
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
                self._supervise()
            except Exception:
                log.exception("health/supervision pass failed")

    async def probe_all(self):
        await asyncio.gather(*(self._probe(name) for name in self.cfg.models))

    async def _probe(self, name: str):
        m = self.cfg.models[name]
        h = self.health[name]
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
            # Mark up immediately; a recovered backend shouldn't wait out a
            # debounce window. A healthy backend also clears all failure
            # bookkeeping — crash-restart backoff and the autoload circuit —
            # so whatever recovered it (explicit load, external restart) closes
            # the circuit without special-casing each path.
            self.resident.add(name)
            self.latency_ms[name] = (time.monotonic() - t0) * 1000
            h.consecutive_fail = 0
            h.restart_attempts = 0
            h.next_restart_at = 0.0
            h.autoload_fail = 0
            h.autoload_open_until = 0.0
        else:
            # Debounce down-marking: a single probe blip (e.g. the backend was
            # busy on a long generation and didn't answer within probe_timeout_s)
            # shouldn't yank it out of the resident set and cause routing to flap.
            h.consecutive_fail += 1
            if h.consecutive_fail >= self.cfg.health.fail_threshold:
                self.resident.discard(name)
                self.latency_ms.pop(name, None)

    def _supervise(self):
        """Watchdog: relaunch managed backends that crashed on their own.
        Skips a crashed backend whose memory has since been taken by a
        conflicting resident model — its slot isn't free, so the correct
        recovery is lazy (the next request for it runs the normal
        unload-then-load path). Runs synchronously; launch() is non-blocking."""
        if self.log_dir is None:
            return
        now = time.monotonic()
        for name, proc in list(self.procs.items()):
            if proc.poll() is None:
                continue  # still alive
            # terminate() pops from self.procs, so anything still here that has
            # exited crashed unexpectedly (we didn't stop it on purpose).
            m = self.cfg.models[name]
            code = proc.returncode
            if not m.restart:
                self.procs.pop(name, None)
                self.active_profiles.pop(name, None)
                self.resident.discard(name)
                log.warning("%s exited (code %s); restart disabled — leaving down", name, code)
                continue
            profile = self.active_profiles.get(name)
            conflicts = self._conflicts_for_load(name, profile)
            if conflicts:
                self.procs.pop(name, None)
                self.active_profiles.pop(name, None)
                self.resident.discard(name)
                log.info("%s crashed (code %s) but %s now occupy its memory; "
                         "deferring recovery to the next request for it", name, code, conflicts)
                continue
            h = self.health[name]
            if now < h.next_restart_at:
                continue  # in backoff window
            h.restart_attempts += 1
            backoff = min(
                self.cfg.health.restart_backoff_s * (2 ** (h.restart_attempts - 1)),
                self.cfg.health.restart_max_backoff_s,
            )
            h.next_restart_at = now + backoff
            self.procs.pop(name, None)
            self.resident.discard(name)
            log.warning("%s crashed (code %s); auto-restarting (attempt %d, next backoff %.0fs)",
                        name, code, h.restart_attempts, backoff)
            try:
                self.launch(name, self.log_dir, profile)
            except Exception:
                log.exception("auto-restart of %s failed", name)

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

    async def ensure_resident(self, name: str, log_dir: Path, profile: str | None = None) -> bool:
        """First-request autoload: launch (idempotent) + block until ready,
        serialized per model so concurrent requests for the same cold model
        wait on one launch instead of racing duplicate subprocesses.

        A circuit breaker guards against a fundamentally-broken backend: after
        `autoload_max_failures` consecutive failed loads, the circuit opens and
        subsequent calls fast-fail for `autoload_cooldown_s` instead of each
        eating a full `load_timeout_s` — otherwise a dead backend makes every
        dashboard request hang for a minute."""
        m = self.cfg.models[name]
        try:
            requested_profile = self.cfg.profile_for(name, profile) if profile else None
        except ValueError:
            return False
        if not m.enabled or not m.autoload:
            return False
        h = self.health[name]
        if time.monotonic() < h.autoload_open_until:
            return False  # circuit open — fast-fail
        lock = self._launch_locks.setdefault(name, asyncio.Lock())
        async with lock:
            if name in self.resident:
                # An ordinary routed request can use whichever profile is
                # already warm. An explicit OpenAI model id such as
                # "ds4-full" is a profile pin: switch only when the current
                # managed process is idle, never silently serve the wrong one.
                if requested_profile is None or self.active_profiles.get(name) == requested_profile.name:
                    return True
                if name not in self.procs or self.procs[name].poll() is not None:
                    return False  # externally started: profile is unknowable/unsafe to replace
                if not await self.wait_idle(name, 0.0):
                    return False
                self.terminate(name)
            selected = requested_profile or self.cfg.profile_for(name)
            selected_name = selected.name if selected else None
            if not self._launch_cmd(name, selected_name):
                return False
            try:
                await self.launch_for_load(name, log_dir, selected_name)
            except ValueError:
                log.exception("autoload failed for %s", name)
                self._record_autoload_result(name, ok=False)
                return False
            ok = await self.wait_resident(name, m.load_timeout_s)
            self._record_autoload_result(name, ok=ok)
            return ok

    def _record_autoload_result(self, name: str, *, ok: bool) -> None:
        h = self.health[name]
        if ok:
            h.autoload_fail = 0
            h.autoload_open_until = 0.0
            return
        h.autoload_fail += 1
        if h.autoload_fail >= self.cfg.health.autoload_max_failures:
            h.autoload_open_until = time.monotonic() + self.cfg.health.autoload_cooldown_s
            log.warning("%s autoload failed %d× — opening circuit for %.0fs (fast-fail)",
                        name, h.autoload_fail, self.cfg.health.autoload_cooldown_s)

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

    def _launch_cmd(self, name: str, profile: str | None = None) -> str | None:
        p = self.cfg.profile_for(name, profile)
        return p.launch_cmd if p else self.cfg.models[name].launch_cmd

    def _conflict_names(self, name: str, profile: str | None = None) -> list[str]:
        p = self.cfg.profile_for(name, profile)
        return list(p.unload_before_load if p else self.cfg.models[name].unload_before_load)

    def _conflicts_for_load(self, name: str, profile: str | None = None) -> list[str]:
        """Return resident backends incompatible with this launch.

        Conflict declarations are interpreted symmetrically: if a resident
        full-residency profile says it cannot coexist with Ornith, loading
        Ornith later must respect that too. This prevents order-dependent OOMs.
        """
        declared = self._conflict_names(name, profile)
        conflicts = [other for other in declared if other in self.resident and other != name]
        for other in sorted(self.resident):
            if other == name:
                continue
            if (name in self._conflict_names(other, self.active_profiles.get(other))
                    and other not in conflicts):
                conflicts.append(other)
        return conflicts

    async def _unload_before_load(self, name: str, profile: str | None = None):
        for other in self._conflicts_for_load(name, profile):
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

    def launch(self, name: str, log_dir: Path, profile: str | None = None) -> str:
        launch_cmd = self._launch_cmd(name, profile)
        profile_label = profile or "default"
        if not launch_cmd:
            raise ValueError(f"{name} has no launch command configured")
        proc = self.procs.get(name)
        if proc and proc.poll() is None:
            return f"{name} already launched (pid {proc.pid}, profile {self.active_profiles.get(name, 'external')})"
        log_dir.mkdir(parents=True, exist_ok=True)
        logfile = open(log_dir / f"{profile or name}.log", "ab")
        proc = subprocess.Popen(
            launch_cmd, shell=True, start_new_session=True,
            stdout=logfile, stderr=subprocess.STDOUT,
        )
        self.procs[name] = proc
        if profile:
            self.active_profiles[name] = profile
        else:
            self.active_profiles.pop(name, None)
        log.info("launched %s profile=%s pid=%d", name, profile_label, proc.pid)
        return f"launched {name} profile {profile_label} (pid {proc.pid}); resident once its /v1/models responds"

    async def launch_for_load(self, name: str, log_dir: Path, profile: str | None = None) -> str:
        selected = self.cfg.profile_for(name, profile)
        selected_name = selected.name if selected else None
        if name in self.resident:
            active = self.active_profiles.get(name)
            if profile is not None and profile != active:
                raise ValueError(
                    f"{name} is already resident with profile {active or 'external'}; unload it before switching profiles"
                )
            return f"{name} already resident"
        await self._unload_before_load(name, selected_name)
        return self.launch(name, log_dir, selected_name)

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
        self.active_profiles.pop(name, None)
        self.resident.discard(name)
        return f"sent SIGTERM to {name} process group (pid {proc.pid})"

    def status(self) -> dict:
        out = {}
        now = time.monotonic()
        for name, m in self.cfg.models.items():
            proc = self.procs.get(name)
            h = self.health[name]
            entry = {
                "resident": name in self.resident,
                "upstream": m.upstream,
                "probe_latency_ms": round(self.latency_ms[name], 1) if name in self.latency_ms else None,
                "managed_pid": proc.pid if proc and proc.poll() is None else None,
                "active_requests": self.active.get(name, 0),
                "tags": m.tags,
                "default_profile": m.default_profile,
                "active_profile": self.active_profiles.get(name),
                "launch_profiles": self.cfg.profiles_for(name),
            }
            # Only surface trouble signals when they're actually active, so a
            # healthy backend's status stays clean/scannable.
            if h.consecutive_fail:
                entry["consecutive_probe_failures"] = h.consecutive_fail
            if h.restart_attempts:
                entry["restart_attempts"] = h.restart_attempts
            if now < h.next_restart_at:
                entry["restart_backoff_s"] = round(h.next_restart_at - now, 1)
            if now < h.autoload_open_until:
                entry["autoload_circuit_open_s"] = round(h.autoload_open_until - now, 1)
            out[name] = entry
        return out
