"""Agent tool implementations: read_file, write_file, list_dir, run_shell.

No path sandboxing - full access, matching what's already been run safely
in practice. run_shell's timeout is a dead-man's-switch (catches a hung
subprocess, which the reasoning-stream supervisor can't see at all, since
nothing is generating while a tool call blocks), not a routine restriction.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from ._procutil import ProcessTimeout, run_shell as _run_shell_proc

MAX_TOOL_OUTPUT_CHARS = 50_000

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file, relative to your working directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path, relative to your working directory."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file, creating it (and any parent directories) if needed, "
                           "or overwriting it if it already exists.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path, relative to your working directory."},
                    "content": {"type": "string", "description": "Full file content to write."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List the contents of a directory, relative to your working directory. "
                           "Defaults to the working directory root.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path, relative to your working directory. Defaults to '.'."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a shell command in your working directory and return its combined output. "
                           "Long-running commands (dev servers, watch mode, anything that doesn't exit on its "
                           "own) will be killed after a generous timeout - do not start something you expect "
                           "to keep running.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
]


READ_ONLY_TOOL_SCHEMAS = [t for t in TOOL_SCHEMAS if t["function"]["name"] in ("read_file", "list_dir")]


def _truncate(text: str) -> str:
    if len(text) <= MAX_TOOL_OUTPUT_CHARS:
        return text
    return text[:MAX_TOOL_OUTPUT_CHARS] + f"\n...[truncated, {len(text) - MAX_TOOL_OUTPUT_CHARS} more chars]"


async def _read_file(cwd: Path, path: str) -> str:
    if not path:
        return "error: path is required"
    target = cwd / path
    try:
        content = await asyncio.to_thread(target.read_text, errors="replace")
    except FileNotFoundError:
        return f"error: {path} does not exist"
    except IsADirectoryError:
        return f"error: {path} is a directory, not a file"
    except OSError as e:
        return f"error: could not read {path}: {e}"
    return _truncate(content)


async def _write_file(cwd: Path, path: str, content: str) -> str:
    if not path:
        return "error: path is required"
    target = cwd / path

    def _write():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)

    try:
        await asyncio.to_thread(_write)
    except OSError as e:
        return f"error: could not write {path}: {e}"
    return f"wrote {len(content)} chars to {path}"


async def _list_dir(cwd: Path, path: str) -> str:
    target = cwd / (path or ".")

    def _list() -> str:
        if not target.exists():
            return f"error: {path or '.'} does not exist"
        if not target.is_dir():
            return f"error: {path or '.'} is not a directory"
        entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name))
        lines = [f"{'d' if e.is_dir() else 'f'}  {e.name}" for e in entries]
        return "\n".join(lines) if lines else "(empty directory)"

    return _truncate(await asyncio.to_thread(_list))


async def _run_shell(cwd: Path, command: str, timeout_s: float) -> str:
    if not command:
        return "error: command is required"
    code, out, err = await _run_shell_proc(command, cwd=cwd, timeout_s=timeout_s)
    combined = out
    if err:
        combined += ("\n--- stderr ---\n" if combined else "") + err
    return _truncate(f"exit code: {code}\n{combined}")


async def execute_tool(name: str, arguments: dict, cwd: Path, run_shell_timeout_s: float) -> str:
    """Dispatch and execute one tool call, returning the string to send back
    as the tool result message. ProcessTimeout is allowed to propagate from
    run_shell only - the caller treats that as a distinct terminal condition
    (the dead-man's-switch), not a normal tool error to hand back to the model."""
    if name == "read_file":
        return await _read_file(cwd, arguments.get("path", ""))
    if name == "write_file":
        return await _write_file(cwd, arguments.get("path", ""), arguments.get("content", ""))
    if name == "list_dir":
        return await _list_dir(cwd, arguments.get("path", "."))
    if name == "run_shell":
        return await _run_shell(cwd, arguments.get("command", ""), run_shell_timeout_s)
    return f"error: unknown tool {name!r}"


__all__ = ["TOOL_SCHEMAS", "READ_ONLY_TOOL_SCHEMAS", "execute_tool", "ProcessTimeout"]
