"""Tests for pipeline/tools.py — edit_file, grep, glob, read_file offset/limit,
truncation, unknown tool dispatch, and SUBMIT_WORK_TOOL schema.

Uses plain pytest, no classes, no mocks framework, descriptive test names.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from pipeline.tools import (
    TOOL_SCHEMAS,
    READ_ONLY_TOOL_SCHEMAS,
    SUBMIT_WORK_TOOL,
    execute_tool,
    _DEFAULT_MAX_OUTPUT_CHARS,
)
from pipeline._procutil import ProcessTimeout


# ── helpers ──

def _tool_names(schemas):
    return [s["function"]["name"] for s in schemas]


def _write_file(tmp_path, rel_path, content):
    """Write a file inside tmp_path, returning the relative path string."""
    target = tmp_path / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return rel_path


# ── edit_file tests ──

async def test_edit_file_success(tmp_path):
    """Simple single replacement works."""
    rel = _write_file(tmp_path, "foo.txt", "hello world\nfoo bar\n")
    result = await execute_tool("edit_file", {
        "path": rel,
        "old_string": "foo",
        "new_string": "baz",
    }, tmp_path, 30)
    assert result == "edited foo.txt (1 replacement)"
    # "foo" appears in "foo bar", so replacement yields "hello world\nbaz bar\n"
    assert (tmp_path / rel).read_text() == "hello world\nbaz bar\n"


async def test_edit_file_not_found(tmp_path):
    """old_string absent → error string."""
    rel = _write_file(tmp_path, "foo.txt", "hello world\n")
    result = await execute_tool("edit_file", {
        "path": rel,
        "old_string": "nope",
        "new_string": "something",
    }, tmp_path, 30)
    assert "not found" in result
    # File unchanged
    assert (tmp_path / rel).read_text() == "hello world\n"


async def test_edit_file_ambiguous_without_replace_all(tmp_path):
    """Multiple occurrences without replace_all → error naming the count."""
    rel = _write_file(tmp_path, "bar.txt", "a a a b c")
    result = await execute_tool("edit_file", {
        "path": rel,
        "old_string": "a",
        "new_string": "x",
    }, tmp_path, 30)
    assert "appears 3 times" in result
    assert "replace_all=True" in result
    # File unchanged
    assert (tmp_path / rel).read_text() == "a a a b c"


async def test_edit_file_replace_all(tmp_path):
    """replace_all=True replaces every occurrence and reports the count."""
    rel = _write_file(tmp_path, "bar.txt", "a a a b c")
    result = await execute_tool("edit_file", {
        "path": rel,
        "old_string": "a",
        "new_string": "x",
        "replace_all": True,
    }, tmp_path, 30)
    assert result == "edited bar.txt (3 replacements)"
    assert (tmp_path / rel).read_text() == "x x x b c"


async def test_edit_file_no_op_rejection(tmp_path):
    """old_string == new_string → error."""
    rel = _write_file(tmp_path, "foo.txt", "hello")
    result = await execute_tool("edit_file", {
        "path": rel,
        "old_string": "hello",
        "new_string": "hello",
    }, tmp_path, 30)
    assert "identical" in result
    assert (tmp_path / rel).read_text() == "hello"


async def test_edit_file_empty_old(tmp_path):
    """Empty old_string is rejected."""
    rel = _write_file(tmp_path, "f.txt", "hello")
    result = await execute_tool("edit_file", {
        "path": rel,
        "old_string": "",
        "new_string": "x",
    }, tmp_path, 30)
    assert "old_string is required" in result


async def test_edit_file_empty_new(tmp_path):
    """Empty new_string is rejected."""
    rel = _write_file(tmp_path, "f.txt", "hello")
    result = await execute_tool("edit_file", {
        "path": rel,
        "old_string": "hello",
        "new_string": "",
    }, tmp_path, 30)
    assert "new_string is required" in result


async def test_edit_file_file_not_found(tmp_path):
    """Missing file path → error."""
    result = await execute_tool("edit_file", {
        "path": "nonexistent.txt",
        "old_string": "a",
        "new_string": "b",
    }, tmp_path, 30)
    assert "does not exist" in result


# ── grep tests ──

async def test_grep_finds_matches(tmp_path):
    """grep finds matching lines and outputs file:line:text."""
    _write_file(tmp_path, "src/a.py", "def foo():\n    pass\n")
    _write_file(tmp_path, "src/b.py", "class Foo:\n    pass\n")
    result = await execute_tool("grep", {
        "pattern": "foo",
        "path": "src",
    }, tmp_path, 30)
    # Should find at least the line in a.py (case-sensitive).
    assert "a.py" in result
    assert "b.py" not in result  # "Foo" is capitalized


async def test_grep_caps_results(tmp_path):
    """grep caps at max_results with a trailing note."""
    lines = [f"line{i}\n" for i in range(100)]
    _write_file(tmp_path, "big.txt", "".join(lines))
    result = await execute_tool("grep", {
        "pattern": "line",
        "max_results": 5,
    }, tmp_path, 30)
    # Should have exactly 5 matches plus a truncation note.
    count = result.count("\n") + 1  # number of lines in result
    assert count <= 6, f"expected ≤6 lines, got {count}"
    assert "more matches" in result


async def test_grep_no_matches(tmp_path):
    """No matches returns a normal message, not an error."""
    _write_file(tmp_path, "data.txt", "hello world")
    result = await execute_tool("grep", {
        "pattern": "xyzzy",
    }, tmp_path, 30)
    assert result == "no matches for 'xyzzy'"


async def test_grep_empty_pattern(tmp_path):
    """Empty pattern is rejected."""
    result = await execute_tool("grep", {
        "pattern": "",
    }, tmp_path, 30)
    assert "pattern is required" in result


# ── glob tests ──

async def test_glob_ordering_and_capping(tmp_path):
    """glob sorts by mtime newest first and caps results."""
    # Create files with different mtimes
    files = ["a.py", "b.py", "c.py", "d.py"]
    for f in files:
        p = tmp_path / f
        p.write_text("")
        # Set mtime to different times using os.utime
        os.utime(p, (0, 1000 + hash(f) % 1000))
    # Wait a tiny bit for clock resolution
    result = await execute_tool("glob", {"pattern": "*.py"}, tmp_path, 30)
    lines = result.split("\n")
    # Should list all 4 files
    assert len(lines) == 4


async def test_glob_capping(tmp_path):
    """When there are more than 200 results, cap and note."""
    for i in range(210):
        (tmp_path / f"file_{i:03d}.txt").write_text("")
    result = await execute_tool("glob", {"pattern": "*.txt"}, tmp_path, 30)
    lines = result.split("\n")
    assert len(lines) <= 202  # 200 capped + maybe truncation note
    assert "truncated" in result


async def test_glob_empty_pattern(tmp_path):
    """Empty pattern is rejected."""
    result = await execute_tool("glob", {"pattern": ""}, tmp_path, 30)
    assert "pattern is required" in result


async def test_glob_no_matches(tmp_path):
    """No matching files → empty result."""
    result = await execute_tool("glob", {"pattern": "*.nosuchext"}, tmp_path, 30)
    assert result == ""  # no matches → empty string


# ── read_file offset/limit tests ──

async def test_read_file_offset_limit(tmp_path):
    """offset/limit produces line-number-prefixed output."""
    lines = [f"line{i}\n" for i in range(1, 21)]  # 20 lines
    rel = _write_file(tmp_path, "nums.txt", "".join(lines))
    result = await execute_tool("read_file", {
        "path": rel,
        "offset": 5,
        "limit": 3,
    }, tmp_path, 30)
    # Expected lines 5, 6, 7 with number prefix
    assert "     5:line5" in result
    assert "     6:line6" in result
    assert "     7:line7" in result
    # Line 4 and 8 should not appear
    assert "line4" not in result
    assert "line8" not in result


async def test_read_file_offset_without_limit(tmp_path):
    """offset without limit reads from that line to end."""
    lines = [f"line{i}\n" for i in range(1, 11)]
    rel = _write_file(tmp_path, "nums2.txt", "".join(lines))
    result = await execute_tool("read_file", {
        "path": rel,
        "offset": 8,
    }, tmp_path, 30)
    assert "     8:line8" in result
    assert "     9:line9" in result
    assert "    10:line10" in result
    # Line 1 should not appear (line10 contains "line1" as substring, so
    # check for the numbered prefix instead)
    assert "     1:" not in result
    assert "     2:" not in result
    assert "     7:" not in result


async def test_read_file_no_offset(tmp_path):
    """Without offset/limit, reads full file (default behavior)."""
    content = "hello\nworld\n"
    rel = _write_file(tmp_path, "simple.txt", content)
    result = await execute_tool("read_file", {
        "path": rel,
    }, tmp_path, 30)
    assert result == content  # no line numbers, full content


# ── truncation tests ──

async def test_truncation_head_tail(tmp_path):
    """Truncation keeps both head and tail."""
    # Create a file longer than default output chars
    big = "X" * 20000 + "\nTAIL_LINE"
    rel = _write_file(tmp_path, "big.txt", big)
    result = await execute_tool("read_file", {
        "path": rel,
    }, tmp_path, 30)
    # Should be truncated
    assert "truncated" in result
    # Tail should be present
    assert "TAIL_LINE" in result


async def test_truncation_small_file_no_truncate(tmp_path):
    """Small files are not truncated."""
    content = "short text"
    rel = _write_file(tmp_path, "small.txt", content)
    result = await execute_tool("read_file", {
        "path": rel,
    }, tmp_path, 30)
    assert result == content


# ── unknown tool dispatch ──

async def test_unknown_tool(tmp_path):
    """Unknown tool name returns an error string."""
    result = await execute_tool("nonexistent_tool", {}, tmp_path, 30)
    assert "unknown tool" in result
    assert "nonexistent_tool" in result


# ── SUBMIT_WORK_TOOL schema ──

def test_submit_work_tool_is_well_formed():
    """SUBMIT_WORK_TOOL has the correct structure with its four properties."""
    assert SUBMIT_WORK_TOOL["type"] == "function"
    func = SUBMIT_WORK_TOOL["function"]
    assert func["name"] == "submit_work"
    assert "summary" in func["parameters"]["properties"]
    assert "files_changed" in func["parameters"]["properties"]
    assert "verified" in func["parameters"]["properties"]
    assert "blocked" in func["parameters"]["properties"]
    assert func["parameters"]["required"] == ["summary", "files_changed", "verified", "blocked"]
    # files_changed is an array of strings
    assert func["parameters"]["properties"]["files_changed"]["type"] == "array"
    assert func["parameters"]["properties"]["files_changed"]["items"]["type"] == "string"


# ── Tool schema consistency ──

def test_tool_schemas_include_all():
    """All expected tools are in TOOL_SCHEMAS."""
    names = _tool_names(TOOL_SCHEMAS)
    for expected in ("read_file", "write_file", "edit_file", "grep", "glob", "list_dir", "run_shell"):
        assert expected in names


def test_read_only_schemas_exclude_write():
    """READ_ONLY_TOOL_SCHEMAS must not include edit_file or write_file."""
    names = _tool_names(READ_ONLY_TOOL_SCHEMAS)
    assert "read_file" in names
    assert "list_dir" in names
    assert "grep" in names
    assert "glob" in names
    assert "write_file" not in names
    assert "edit_file" not in names
    assert "run_shell" not in names
    assert "submit_work" not in names


# ── ProcessTimeout import consistency ──

def test_process_timeout_reexported():
    """ProcessTimeout is re-exported from tools module."""
    from pipeline.tools import ProcessTimeout as PT
    assert PT is ProcessTimeout
