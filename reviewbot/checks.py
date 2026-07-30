"""Predefined checks for reviewbot.

Each check is a callable that takes a FileDiff and returns a list of Finding
objects.  The module also exports ALL_CHECKS (a set of check names) and
run_checks() which produces a Review from a list of diffs.
"""
from __future__ import annotations

from reviewbot.models import FileDiff, Finding, Review, Severity

# ---------------------------------------------------------------------------
# Check helpers
# ---------------------------------------------------------------------------

def _only_py(filediff: FileDiff) -> bool:
    """Return True if *filediff* ends with ``.py``."""
    return filediff.path.endswith('.py')


def _make_finding(
    filediff: FileDiff,
    line: int,
    severity: Severity,
    check: str,
    message: str,
) -> Finding:
    return Finding(
        path=filediff.path,
        line=line,
        severity=severity,
        check=check,
        message=message,
    )

# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def debug_print(filediff: FileDiff) -> list[Finding]:
    """WARNING for every added line that contains a bare ``print(`` call."""
    if not _only_py(filediff):
        return []
    findings: list[Finding] = []
    for line_no, text in filediff.added_lines:
        if 'print(' in text:
            findings.append(_make_finding(
                filediff, line_no, Severity.WARNING,
                'debug-print', 'line contains debug print call',
            ))
    return findings


def todo_comment(filediff: FileDiff) -> list[Finding]:
    """INFO for every added line that contains ``TODO`` or ``FIXME``."""
    if not _only_py(filediff):
        return []
    findings: list[Finding] = []
    for line_no, text in filediff.added_lines:
        if 'TODO' in text or 'FIXME' in text:
            findings.append(_make_finding(
                filediff, line_no, Severity.INFO,
                'todo-comment', 'line contains TODO or FIXME',
            ))
    return findings


def bare_except(filediff: FileDiff) -> list[Finding]:
    """ERROR for every added line whose stripped content is exactly ``except:``."""
    if not _only_py(filediff):
        return []
    findings: list[Finding] = []
    for line_no, text in filediff.added_lines:
        if text.strip() == 'except:':
            findings.append(_make_finding(
                filediff, line_no, Severity.ERROR,
                'bare-except', 'bare except clause detected',
            ))
    return findings


def long_line(filediff: FileDiff) -> list[Finding]:
    """WARNING for every added line longer than 100 characters."""
    if not _only_py(filediff):
        return []
    findings: list[Finding] = []
    for line_no, text in filediff.added_lines:
        if len(text) > 100:
            findings.append(_make_finding(
                filediff, line_no, Severity.WARNING,
                'long-line', f'line exceeds 100 characters ({len(text)} chars)',
            ))
    return findings


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Exported for CLI help text and filtering: a set of all known check names.
ALL_CHECKS: set[str] = {
    "debug-print",
    "todo-comment",
    "bare-except",
    "long-line",
}

# Internal mapping from check name to (description, callable).
_CHECK_REGISTRY: dict[str, tuple[str, callable]] = {
    "debug-print": ("WARNING for lines containing print()", debug_print),
    "todo-comment": ("INFO for lines containing TODO or FIXME", todo_comment),
    "bare-except": ("ERROR for bare except:", bare_except),
    "long-line": ("WARNING for lines > 100 characters", long_line),
}


def run_checks(
    filediffs: list[FileDiff],
    enabled: set[str] | None = None,
) -> Review:
    """Run all (or a subset of) checks on *filediffs* and return a Review.

    Parameters
    ----------
    filediffs:
        List of FileDiff objects to inspect.
    enabled:
        If *None*, all checks are run.  Otherwise only checks whose name is
        in *enabled* are run.

    Returns
    -------
    A ``Review`` with all findings collected.
    """
    review = Review()
    review.files_reviewed = len(filediffs)

    if enabled is None:
        enabled = ALL_CHECKS

    checks_to_run = [
        _CHECK_REGISTRY[name]
        for name in enabled
        if name in _CHECK_REGISTRY
    ]

    for filediff in filediffs:
        if not _only_py(filediff):
            continue
        for _desc, func in checks_to_run:
            findings = func(filediff)
            review.findings.extend(findings)

    return review
