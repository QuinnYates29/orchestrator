"""Tests for reviewbot.checks."""
from __future__ import annotations

import pytest

from reviewbot.models import FileDiff, Finding, Hunk, Review, Severity
from reviewbot.checks import (
    ALL_CHECKS,
    debug_print,
    todo_comment,
    bare_except,
    long_line,
    run_checks,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_filediff(path: str, lines: list[str], start_line: int = 1) -> FileDiff:
    """Build a FileDiff with a single hunk containing *lines*."""
    return FileDiff(
        path=path,
        hunks=[Hunk(start_line=start_line, lines=lines)],
    )


# ---------------------------------------------------------------------------
# debug_print
# ---------------------------------------------------------------------------

def test_debug_print_finds_print_call():
    fd = _make_filediff("foo.py", ["x = 1", "print('hello')", "y = 2"])
    findings = debug_print(fd)
    assert len(findings) == 1
    f = findings[0]
    assert f.line == 2
    assert f.severity is Severity.WARNING
    assert f.check == "debug-print"
    assert f.path == "foo.py"


def test_debug_print_ignores_non_py():
    fd = _make_filediff("foo.txt", ["print('hello')"])
    assert debug_print(fd) == []


def test_debug_print_no_match():
    fd = _make_filediff("foo.py", ["x = 1", "y = 2"])
    assert debug_print(fd) == []


# ---------------------------------------------------------------------------
# todo_comment
# ---------------------------------------------------------------------------

def test_todo_comment_finds_todo():
    fd = _make_filediff("bar.py", ["# TODO: fix this", "pass"])
    findings = todo_comment(fd)
    assert len(findings) == 1
    f = findings[0]
    assert f.line == 1
    assert f.severity is Severity.INFO
    assert f.check == "todo-comment"


def test_todo_comment_finds_fixme():
    fd = _make_filediff("bar.py", ["# FIXME: broken", "pass"])
    findings = todo_comment(fd)
    assert len(findings) == 1
    assert findings[0].line == 1


def test_todo_comment_ignores_non_py():
    fd = _make_filediff("bar.txt", ["# TODO: fix"])
    assert todo_comment(fd) == []


def test_todo_comment_no_match():
    fd = _make_filediff("bar.py", ["# nothing here"])
    assert todo_comment(fd) == []


# ---------------------------------------------------------------------------
# bare_except
# ---------------------------------------------------------------------------

def test_bare_except_finds_except():
    fd = _make_filediff("mod.py", ["except:", "    pass"])
    findings = bare_except(fd)
    assert len(findings) == 1
    f = findings[0]
    assert f.line == 1
    assert f.severity is Severity.ERROR
    assert f.check == "bare-except"


def test_bare_except_ignores_except_value():
    fd = _make_filediff("mod.py", ["except ValueError:", "    pass"])
    assert bare_except(fd) == []


def test_bare_except_ignores_non_py():
    fd = _make_filediff("mod.txt", ["except:"])
    assert bare_except(fd) == []


def test_bare_except_no_match():
    fd = _make_filediff("mod.py", ["pass"])
    assert bare_except(fd) == []


def test_bare_except_stripped_line():
    """Whitespace before 'except:' should still match."""
    fd = _make_filediff("mod.py", ["  except:", "    pass"])
    findings = bare_except(fd)
    assert len(findings) == 1
    assert findings[0].line == 1


# ---------------------------------------------------------------------------
# long_line
# ---------------------------------------------------------------------------

def test_long_line_finds_long():
    fd = _make_filediff("util.py", ["a" * 101])
    findings = long_line(fd)
    assert len(findings) == 1
    f = findings[0]
    assert f.line == 1
    assert f.severity is Severity.WARNING
    assert f.check == "long-line"
    assert "101" in f.message


def test_long_line_ignores_short():
    fd = _make_filediff("util.py", ["a" * 100])
    assert long_line(fd) == []


def test_long_line_ignores_non_py():
    fd = _make_filediff("util.txt", ["a" * 101])
    assert long_line(fd) == []


def test_long_line_exact_boundary():
    """Exactly 100 chars is OK, 101 is not."""
    fd = _make_filediff("util.py", ["a" * 100, "b" * 101])
    findings = long_line(fd)
    assert len(findings) == 1
    assert findings[0].line == 2


# ---------------------------------------------------------------------------
# ALL_CHECKS
# ---------------------------------------------------------------------------

def test_all_checks_has_expected_names():
    names = [name for name, desc, func in ALL_CHECKS]
    assert names == ["debug-print", "todo-comment", "bare-except", "long-line"]


def test_all_checks_functions_are_callable():
    for name, desc, func in ALL_CHECKS:
        assert callable(func)


# ---------------------------------------------------------------------------
# run_checks
# ---------------------------------------------------------------------------

def test_run_checks_all():
    diffs = [
        _make_filediff("a.py", ["print('x')"]),
        _make_filediff("b.py", ["TODO"]),
        _make_filediff("c.py", ["except:"]),
        _make_filediff("d.py", ["a" * 101]),
    ]
    review = run_checks(diffs)
    assert review.files_reviewed == 4
    # Each check should fire once per matching file
    assert len(review.findings) == 4


def test_run_checks_filtered():
    diffs = [
        _make_filediff("a.py", ["print('x')", "TODO"]),
    ]
    review = run_checks(diffs, enabled={"debug-print"})
    assert review.files_reviewed == 1
    assert len(review.findings) == 1
    assert review.findings[0].check == "debug-print"


def test_run_checks_empty_diffs():
    review = run_checks([])
    assert review.files_reviewed == 0
    assert review.findings == []


def test_run_checks_skips_non_py():
    diffs = [
        _make_filediff("a.py", ["print('x')"]),
        _make_filediff("b.txt", ["print('y')"]),
    ]
    review = run_checks(diffs)
    assert review.files_reviewed == 2
    assert len(review.findings) == 1  # only the .py file


def test_run_checks_none_enabled_runs_all():
    diffs = [
        _make_filediff("a.py", ["print('x')"]),
    ]
    review = run_checks(diffs, enabled=None)
    assert len(review.findings) == 1


def test_run_checks_empty_enabled_runs_none():
    diffs = [
        _make_filediff("a.py", ["print('x')"]),
    ]
    review = run_checks(diffs, enabled=set())
    assert review.findings == []
