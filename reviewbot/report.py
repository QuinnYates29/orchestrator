"""Formatting functions for Review objects."""
from __future__ import annotations

import json
from typing import Any

from reviewbot.models import Finding, Review, Severity


def _severity_str(severity: Severity) -> str:
    """Return the human-readable severity string for a Severity enum value."""
    return severity.value  # "info", "warning", "error"


def _line_text(finding: Finding) -> str:
    """Format a single finding as 'path:line: SEVERITY [check] message'."""
    sev = _severity_str(finding.severity)
    return f"{finding.path}:{finding.line}: {sev} [{finding.check}] {finding.message}"


def format_text(review: Review) -> str:
    """Format a Review into a human-readable text report.

    Findings are grouped by file (sorted by path), sorted by line within each
    file. The report ends with a summary line.

    Parameters
    ----------
    review : Review
        The review to format.

    Returns
    -------
    str
        Formatted text report.
    """
    if not review.findings:
        return "No findings."

    # Group findings by file
    grouped: dict[str, list[Finding]] = {}
    for f in review.findings:
        grouped.setdefault(f.path, []).append(f)

    # Sort files by path
    lines: list[str] = []
    for path in sorted(grouped):
        file_findings = grouped[path]
        # Sort within file by line number
        for finding in sorted(file_findings, key=lambda x: x.line):
            lines.append(_line_text(finding))

    # Summary line
    num_files = len(grouped)
    num_findings = len(review.findings)
    if num_findings == 1:
        lines.append(f"1 finding across {num_files} file(s)")
    else:
        lines.append(f"{num_findings} finding(s) across {num_files} file(s)")

    return "\n".join(lines)


def format_json(review: Review) -> str:
    """Format a Review into a JSON string.

    The JSON object has:
      - "findings": list of dicts with keys path, line, severity, check, message
      - "files_reviewed": integer

    Parameters
    ----------
    review : Review
        The review to format.

    Returns
    -------
    str
        JSON string with indent=2.
    """
    findings_list: list[dict[str, Any]] = []
    for f in review.findings:
        findings_list.append({
            "path": f.path,
            "line": f.line,
            "severity": _severity_str(f.severity),
            "check": f.check,
            "message": f.message,
        })

    obj: dict[str, Any] = {
        "findings": findings_list,
        "files_reviewed": review.files_reviewed,
    }

    return json.dumps(obj, indent=2)
