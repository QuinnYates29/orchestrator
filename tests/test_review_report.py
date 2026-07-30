"""Tests for pipeline/review/report.py."""
from __future__ import annotations

import json

import pytest

from pipeline.review.models import Finding, Review, Severity
from pipeline.review.report import format_text, format_json


def test_format_text_empty():
    review = Review(findings=[], files_reviewed=2)
    assert format_text(review) == "No findings."


def test_format_text_single_finding():
    findings = [
        Finding(path="a.py", line=10, severity=Severity.ERROR,
                check="my-check", message="bad thing"),
    ]
    review = Review(findings=findings, files_reviewed=1)
    expected = (
        "a.py:10: error [my-check] bad thing\n"
        "1 finding in 1 file (1 reviewed)"
    )
    assert format_text(review) == expected


def test_format_text_multiple_files():
    findings = [
        Finding(path="b.py", line=5, severity=Severity.WARNING,
                check="lint", message="unused var"),
        Finding(path="a.py", line=2, severity=Severity.INFO,
                check="fmt", message="missing space"),
        Finding(path="a.py", line=1, severity=Severity.ERROR,
                check="lint", message="syntax"),
    ]
    review = Review(findings=findings, files_reviewed=2)
    result = format_text(review)
    lines = result.split("\n")
    assert lines[0] == "a.py:1: error [lint] syntax"
    assert lines[1] == "a.py:2: info [fmt] missing space"
    assert lines[2] == "b.py:5: warning [lint] unused var"
    assert lines[3].startswith("3 findings in 2 files")


def test_format_text_sorted_by_file_then_line():
    findings = [
        Finding(path="z.py", line=10, severity=Severity.INFO,
                check="a", message="z"),
        Finding(path="a.py", line=5, severity=Severity.INFO,
                check="b", message="a"),
        Finding(path="a.py", line=1, severity=Severity.INFO,
                check="c", message="a"),
    ]
    review = Review(findings=findings, files_reviewed=2)
    result = format_text(review)
    lines = result.split("\n")
    assert lines[0] == "a.py:1: info [c] a"
    assert lines[1] == "a.py:5: info [b] a"
    assert lines[2] == "z.py:10: info [a] z"
    assert lines[3].startswith("3 findings in 2 files")


def test_format_text_single_file_single_finding():
    findings = [
        Finding(path="x.py", line=1, severity=Severity.ERROR,
                check="err", message="error here"),
    ]
    review = Review(findings=findings, files_reviewed=1)
    result = format_text(review)
    assert result.endswith("1 finding in 1 file (1 reviewed)")


def test_format_json_empty():
    review = Review(findings=[], files_reviewed=0)
    result = format_json(review)
    parsed = json.loads(result)
    assert parsed == {"findings": [], "files_reviewed": 0}


def test_format_json():
    findings = [
        Finding(path="a.py", line=10, severity=Severity.ERROR,
                check="my-check", message="bad thing"),
        Finding(path="b.py", line=5, severity=Severity.WARNING,
                check="lint", message="unused var"),
    ]
    review = Review(findings=findings, files_reviewed=2)
    result = format_json(review)
    parsed = json.loads(result)
    assert parsed["files_reviewed"] == 2
    assert len(parsed["findings"]) == 2
    f0 = parsed["findings"][0]
    assert f0["path"] == "a.py"
    assert f0["line"] == 10
    assert f0["severity"] == "error"
    assert f0["check"] == "my-check"
    assert f0["message"] == "bad thing"
    f1 = parsed["findings"][1]
    assert f1["path"] == "b.py"
    assert f1["line"] == 5
    assert f1["severity"] == "warning"
    assert f1["check"] == "lint"
    assert f1["message"] == "unused var"


def test_format_json_indent():
    findings = [
        Finding(path="a.py", line=1, severity=Severity.INFO,
                check="c", message="m"),
    ]
    review = Review(findings=findings, files_reviewed=1)
    result = format_json(review)
    # Ensure it's pretty-printed with 2-space indent
    parsed = json.loads(result)
    assert parsed["files_reviewed"] == 1
    # Check that the output contains newlines indicating indentation
    assert "\n" in result
    # Re-encode the expected structure and compare
    expected = json.dumps({"findings": [
        {"path": "a.py", "line": 1, "severity": "info",
         "check": "c", "message": "m", "source": "static"}
    ], "files_reviewed": 1}, indent=2)
    assert result == expected
