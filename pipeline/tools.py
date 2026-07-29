"""Agent tool implementations: read_file, write_file, list_dir, run_shell,
edit_file, grep, glob.

No path sandboxing - full access, matching what's already been run safely
in practice. run_shell's timeout is a dead-man's-switch (catches a hung
subprocess, which the reasoning-stream supervisor can't see at all, since
nothing is generating while a tool call blocks), not a routine restriction.

Tools that mutate files (write_file, edit_file) are intentionally omitted
from READ_ONLY_TOOL_SCHEMAS so that read-only agents don't get offered
write capability.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from ._procutil import ProcessTimeout, run_argv, run_shell as _run_shell_proc

# Default cap — large enough to hold a typical test failure summary but not
# so large that it crowds context on every subsequent turn. Callers can
# override via the `max_output_chars` parameter of execute_tool.
_DEFAULT_MAX_OUTPUT_CHARS = 8000

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file, relative to your working directory. "
                           "Optionally read a specific line range (1-based offset, line count limit). "
                           "When offset/limit are given, each output line is prefixed with its line number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path, relative to your working directory."},
                    "offset": {"type": "integer", "description": "1-based starting line number (optional)."},
                    "limit": {"type": "integer", "description": "Number of lines to read (optional)."},
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
            "name": "edit_file",
            "description": "Exact string replacement in a file. "
                           "Replaces old_string with new_string. "
                           "If old_string appears multiple times and replace_all is false, returns an error. "
                           "With replace_all=True, replaces every occurrence. "
                           "old_string == new_string is an error (no-op).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path, relative to your working directory."},
                    "old_string": {"type": "string", "description": "Exact text to find (case-sensitive)."},
                    "new_string": {"type": "string", "description": "Text to replace with."},
                    "replace_all": {"type": "boolean", "description": "Replace all occurrences if true (default false)."},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search for a pattern in files. Uses ripgrep if available, otherwise grep -rn. "
                           "Output is file:line:text, one per line. Results are capped; a trailing note "
                           "is added when truncated.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Search pattern (regex)."},
                    "path": {"type": "string", "description": "Directory or file to search (default '.')."},
                    "glob": {"type": "string", "description": "Optional file glob filter (e.g. '*.py')."},
                    "max_results": {"type": "integer", "description": "Maximum results to return (default 50)."},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Match paths relative to the working directory against a glob pattern. "
                           "Results are sorted by modification time, newest first. "
                           "The result list is capped; a note is added when truncated.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern (e.g. '**/*.py')."},
                },
                "required": ["pattern"],
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


SUBMIT_WORK_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_work",
        "description": "Finalize the current work unit. Must be called exactly once at the end of every "
                       "task, after all file changes are done and verification has been attempted. "
                       "The worker loop interprets this tool call — it is not dispatched by execute_tool.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "What was done in this work unit."},
                "files_changed": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Paths of files that were created or modified.",
                },
                "verified": {
                    "type": "string",
                    "description": "How the work was verified (command run, tests passed, etc.). "
                                   "Must not be assumed; if nothing was run, say so.",
                },
                "blocked": {
                    "type": "string",
                    "description": "Anything that could not be completed. Must be honest and explicit.",
                },
            },
            "required": ["summary", "files_changed", "verified", "blocked"],
        },
    },
}


READ_ONLY_TOOL_SCHEMAS = [t for t in TOOL_SCHEMAS
                          if t["function"]["name"] in ("read_file", "list_dir", "grep", "glob")]


def _truncate(text: str, max_chars: int) -> str:
    """Truncate *text* to *max_chars*, keeping the head and tail with a marker
    in the middle so the model still sees a test failure summary and a stack
    trace's final line."""
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail_start = len(text) - (max_chars - head)
    marker = f"\n...[truncated {len(text) - max_chars} chars]...\n"
    return text[:head] + marker + text[tail_start:]


async def _read_file(cwd: Path, path: str, max_output_chars: int,
                     offset: int | None = None, limit: int | None = None) -> str:
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

    lines = content.split("\n")
    # Convert 1-based offset to 0-based; offset=1 means line 1.
    if offset is not None:
        start = offset - 1
        if start < 0:
            start = 0
        if limit is not None:
            end = start + limit
        else:
            end = len(lines)
        selected = lines[start:end]
        # Line-number-prefix output
        out_lines = []
        for i, line in enumerate(selected, start=start + 1):
            out_lines.append(f"{i:>6}:{line}")
        result = "\n".join(out_lines)
        # When reading a slice, no truncation — the model asked for a specific
        # range and we keep it intact. (The range itself is the cap.)
        return result
    else:
        return _truncate(content, max_output_chars)


async def _write_file(cwd: Path, path: str, content: str, max_output_chars: int) -> str:
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


async def _edit_file(cwd: Path, path: str, old_string: str, new_string: str,
                      replace_all: bool, max_output_chars: int) -> str:
    if not path:
        return "error: path is required"
    if not old_string:
        return "error: old_string is required"
    if not new_string:
        return "error: new_string is required"
    if old_string == new_string:
        return "error: old_string and new_string are identical (no-op edit)"

    target = cwd / path
    try:
        content = await asyncio.to_thread(target.read_text, errors="replace")
    except FileNotFoundError:
        return f"error: {path} does not exist"
    except IsADirectoryError:
        return f"error: {path} is a directory, not a file"
    except OSError as e:
        return f"error: could not read {path}: {e}"

    count = content.count(old_string)
    if count == 0:
        return f"error: {old_string!r} not found in {path}"

    if not replace_all and count > 1:
        return f"error: {old_string!r} appears {count} times in {path}; use replace_all=True to replace all occurrences"

    if replace_all:
        new_content = content.replace(old_string, new_string)
        replacements = count
    else:
        new_content = content.replace(old_string, new_string, 1)
        replacements = 1

    def _write():
        target.write_text(new_content)

    try:
        await asyncio.to_thread(_write)
    except OSError as e:
        return f"error: could not write {path}: {e}"

    return f"edited {path} ({replacements} replacement{'s' if replacements > 1 else ''})"


async def _list_dir(cwd: Path, path: str, max_output_chars: int) -> str:
    target = cwd / (path or ".")

    def _list() -> str:
        if not target.exists():
            return f"error: {path or '.'} does not exist"
        if not target.is_dir():
            return f"error: {path or '.'} is not a directory"
        entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name))
        lines = [f"{'d' if e.is_dir() else 'f'}  {e.name}" for e in entries]
        return "\n".join(lines) if lines else "(empty directory)"

    return _truncate(await asyncio.to_thread(_list), max_output_chars)


async def _run_shell(cwd: Path, command: str, timeout_s: float, max_output_chars: int) -> str:
    if not command:
        return "error: command is required"
    code, out, err = await _run_shell_proc(command, cwd=cwd, timeout_s=timeout_s)
    combined = out
    if err:
        combined += ("\n--- stderr ---\n" if combined else "") + err
    return _truncate(f"exit code: {code}\n{combined}", max_output_chars)


async def _grep(cwd: Path, pattern: str, path: str, glob: str | None,
                 max_results: int, max_output_chars: int) -> str:
    """Search using ripgrep (rg) with fallback to grep -rn. Uses run_argv so
    the pattern is never interpreted by a shell."""
    if not pattern:
        return "error: pattern is required"

    # Prefer ripgrep for speed; fall back to standard grep.
    rg_path = Path("/usr/bin/rg")
    use_rg = rg_path.exists()

    search_dir = cwd / (path or ".")
    argv = []

    if use_rg:
        argv = [str(rg_path)]
        # rg flags: -n for line numbers, --no-heading to avoid file headers,
        # --color never for plain output.
        argv += ["-n", "--no-heading", "--color", "never"]
        if glob:
            argv += ["--glob", glob]
        argv += [pattern, str(search_dir)]
    else:
        argv = ["grep", "-rn"]
        if glob:
            argv += ["--include", glob]
        argv += [pattern, str(search_dir)]

    try:
        code, stdout, stderr = await run_argv(argv, cwd=cwd, timeout_s=30)
    except ProcessTimeout:
        return f"error: grep timed out for {pattern!r}"

    if code != 0 and code != 1:
        # 2 means grep found no pattern match but file errors; rg returns 2 for
        # errors. Treat non-0/1 as error.
        return f"error: grep failed (exit {code}): {stderr.strip()}"

    # grep -rn output: "path:line:text". rg with --no-heading gives the same.
    raw_lines = [l for l in stdout.split("\n") if l.strip()]
    if not raw_lines:
        return f"no matches for {pattern!r}"

    total_matches = len(raw_lines)
    if max_results and total_matches > max_results:
        raw_lines = raw_lines[:max_results]
        raw_lines.append(f"...[{total_matches - max_results} more matches]")

    result = "\n".join(raw_lines)
    return _truncate(result, max_output_chars)


async def _glob(cwd: Path, pattern: str, max_output_chars: int) -> str:
    """Glob paths sorted by modification time (newest first), capped."""
    if not pattern:
        return "error: pattern is required"

    try:
        paths = list(cwd.glob(pattern))
    except Exception as e:
        return f"error: glob failed: {e}"

    # Sort by mtime, newest first. Fall back to name for ties.
    def _mtime(p):
        try:
            return p.stat().st_mtime
        except OSError:
            return 0

    paths.sort(key=lambda p: (_mtime(p), p.name), reverse=True)

    # Cap at 200 results (arbitrary but reasonable).
    MAX_GLOB_RESULTS = 200
    total = len(paths)
    if total > MAX_GLOB_RESULTS:
        paths = paths[:MAX_GLOB_RESULTS]
        note = f"\n...[truncated, {total - MAX_GLOB_RESULTS} more paths]"
    else:
        note = ""

    result = "\n".join(str(p.relative_to(cwd)) for p in paths)
    if note:
        result += note
    return _truncate(result, max_output_chars)


async def execute_tool(name: str, arguments: dict, cwd: Path, run_shell_timeout_s: float,
                       max_output_chars: int = _DEFAULT_MAX_OUTPUT_CHARS) -> str:
    """Dispatch and execute one tool call, returning the string to send back
    as the tool result message. ProcessTimeout is allowed to propagate from
    run_shell only - the caller treats that as a distinct terminal condition
    (the dead-man's-switch), not a normal tool error to hand back to the model.

    max_output_chars is configurable so the agent harness can set it once per
    run (e.g. from LimitsCfg.tool_output_chars) without every caller needing
    to pass it."""
    if name == "read_file":
        return await _read_file(cwd, arguments.get("path", ""), max_output_chars,
                                arguments.get("offset"), arguments.get("limit"))
    if name == "write_file":
        return await _write_file(cwd, arguments.get("path", ""), arguments.get("content", ""), max_output_chars)
    if name == "edit_file":
        return await _edit_file(cwd, arguments.get("path", ""),
                                arguments.get("old_string", ""),
                                arguments.get("new_string", ""),
                                arguments.get("replace_all", False),
                                max_output_chars)
    if name == "grep":
        return await _grep(cwd, arguments.get("pattern", ""),
                           arguments.get("path", "."),
                           arguments.get("glob"),
                           arguments.get("max_results", 50),
                           max_output_chars)
    if name == "glob":
        return await _glob(cwd, arguments.get("pattern", ""), max_output_chars)
    if name == "list_dir":
        return await _list_dir(cwd, arguments.get("path", "."), max_output_chars)
    if name == "run_shell":
        return await _run_shell(cwd, arguments.get("command", ""), run_shell_timeout_s, max_output_chars)
    return f"error: unknown tool {name!r}"


__all__ = [
    "TOOL_SCHEMAS", "READ_ONLY_TOOL_SCHEMAS", "SUBMIT_WORK_TOOL",
    "execute_tool", "ProcessTimeout",
]
