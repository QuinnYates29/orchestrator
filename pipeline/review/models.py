"""Shared data model for pipeline.review.

Every other module in this package depends on these types, so this file is
fixed: it is the contract the parallel pieces are written against. Do not
change these signatures - add to them only if something is genuinely missing,
and never rename a field.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class Hunk:
    """One contiguous changed region within a file."""
    start_line: int              # 1-indexed line number in the NEW file
    lines: list[str] = field(default_factory=list)   # added lines only, no leading '+'


@dataclass
class FileDiff:
    """The changes to a single file in a diff."""
    path: str
    hunks: list[Hunk] = field(default_factory=list)
    is_new: bool = False
    is_deleted: bool = False

    @property
    def added_lines(self) -> list[tuple[int, str]]:
        """Every added line as (line_number, text), across all hunks."""
        out: list[tuple[int, str]] = []
        for hunk in self.hunks:
            for offset, text in enumerate(hunk.lines):
                out.append((hunk.start_line + offset, text))
        return out


@dataclass
class Finding:
    """One thing a check wants to report."""
    path: str
    line: int
    severity: Severity
    check: str                   # the check's short name, e.g. "debug-print"
    message: str


@dataclass
class Review:
    """The full result of reviewing a diff."""
    findings: list[Finding] = field(default_factory=list)
    files_reviewed: int = 0

    @property
    def has_errors(self) -> bool:
        return any(f.severity is Severity.ERROR for f in self.findings)
