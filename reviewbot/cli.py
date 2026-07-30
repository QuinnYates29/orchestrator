"""Command-line interface for reviewbot.

Wire diff parsing, checks, and reporting together with argparse.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from reviewbot.diff import collect_diff, parse_diff
from reviewbot.checks import run_checks, ALL_CHECKS
from reviewbot.report import format_text, format_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build and return the parsed argument namespace."""
    parser = argparse.ArgumentParser(
        description="Review a git diff and report findings."
    )
    parser.add_argument(
        "--staged",
        action="store_true",
        help="Review staged changes (git diff --cached).",
    )
    parser.add_argument(
        "--rev",
        type=str,
        default=None,
        help="Review changes since a given revision (e.g. HEAD~1).",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["text", "json"],
        default="text",
        help="Output format (default: text).",
    )
    parser.add_argument(
        "--check",
        type=str,
        action="append",
        default=None,
        help=(
            "Run only the specified check(s). "
            "Can be repeated to run multiple checks. "
            f"Available: {', '.join(sorted(ALL_CHECKS))}"
        ),
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=".",
        help="Path to the git repository (default: current directory).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns exit code (0 = ok, 1 = has errors)."""
    args = parse_args(argv)

    repo_path = Path(args.repo).resolve()

    # Collect the diff text from git.
    diff_text = collect_diff(
        rev=args.rev,
        staged=args.staged,
        cwd=repo_path,
    )

    # Parse into FileDiff objects.
    diffs = parse_diff(diff_text)

    # Determine which checks to run.
    enabled: set[str] | None = None
    if args.check is not None:
        enabled = set(args.check)

    # Run checks and build the review.
    review = run_checks(diffs, enabled=enabled)

    # Format and print the report.
    if args.format == "json":
        output = format_json(review)
    else:
        output = format_text(review)

    print(output)

    return 1 if review.has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
