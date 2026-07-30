"""Tests for the CLI entry point.

Monkeypatch collect_diff to avoid shelling out to git.
Do not reimplement parsing, checking or formatting logic.
"""
from __future__ import annotations

import pytest
from unittest import mock

from reviewbot.cli import main


def test_main_no_findings_returns_zero(monkeypatch):
    """When the diff has no issues, main returns 0."""
    diff_text = """diff --git a/foo.py b/foo.py
index abc..def 100644
--- a/foo.py
+++ b/foo.py
@@ -1,3 +1,4 @@
 ok
+ok line
"""

    def fake_collect(*, rev=None, staged=False, cwd=None):
        return diff_text

    monkeypatch.setattr("reviewbot.cli.collect_diff", fake_collect)

    exit_code = main(["--format", "text"])
    assert exit_code == 0


def test_main_with_error_finding_returns_one(monkeypatch):
    """When a check finds an ERROR, main returns 1."""
    # A bare "except:" line triggers the bare-except check (ERROR).
    diff_text = """diff --git a/foo.py b/foo.py
index abc..def 100644
--- a/foo.py
+++ b/foo.py
@@ -1,3 +1,4 @@
 ok
+except:
"""

    def fake_collect(*, rev=None, staged=False, cwd=None):
        return diff_text

    monkeypatch.setattr("reviewbot.cli.collect_diff", fake_collect)

    exit_code = main(["--format", "text"])
    assert exit_code == 1


def test_main_staged_flag_passed_to_collect_diff(monkeypatch):
    """The --staged flag is forwarded to collect_diff as staged=True."""
    collected = []

    def fake_collect(*, rev=None, staged=False, cwd=None):
        collected.append((rev, staged, cwd))
        return ""

    monkeypatch.setattr("reviewbot.cli.collect_diff", fake_collect)

    main(["--staged"])
    rev, staged, _ = collected[0]
    assert staged is True
    assert rev is None


def test_main_rev_flag_passed_to_collect_diff(monkeypatch):
    """The --rev argument is forwarded to collect_diff."""
    collected = []

    def fake_collect(*, rev=None, staged=False, cwd=None):
        collected.append((rev, staged, cwd))
        return ""

    monkeypatch.setattr("reviewbot.cli.collect_diff", fake_collect)

    main(["--rev", "HEAD~3"])
    rev, staged, _ = collected[0]
    assert rev == "HEAD~3"
    assert staged is False


def test_main_check_filter_passed_to_run_checks(monkeypatch):
    """--check limits which checks are enabled."""
    collected = []

    def fake_collect(*, rev=None, staged=False, cwd=None):
        return ""

    monkeypatch.setattr("reviewbot.cli.collect_diff", fake_collect)

    # We need to patch run_checks too to inspect the enabled set.
    def fake_run_checks(diffs, enabled=None):
        collected.append(enabled)
        from reviewbot.models import Review
        return Review()

    monkeypatch.setattr("reviewbot.cli.run_checks", fake_run_checks)

    main(["--check", "debug-print", "--check", "long-line"])
    enabled = collected[0]
    assert enabled == {"debug-print", "long-line"}


def test_main_json_format_calls_format_json(monkeypatch):
    """--format json causes format_json to be used."""
    outputs = []

    def fake_collect(*, rev=None, staged=False, cwd=None):
        return ""

    monkeypatch.setattr("reviewbot.cli.collect_diff", fake_collect)

    def fake_format_text(review):
        outputs.append("text")
        return "text output"

    def fake_format_json(review):
        outputs.append("json")
        return '{"json": true}'

    monkeypatch.setattr("reviewbot.cli.format_text", fake_format_text)
    monkeypatch.setattr("reviewbot.cli.format_json", fake_format_json)

    main(["--format", "json"])
    assert outputs == ["json"]
