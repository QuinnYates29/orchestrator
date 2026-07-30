"""Command-line interface for reviewbot.

Wire diff parsing, checks, and reporting together with argparse.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from reviewbot.checks import DEFAULT_MAX_LINE_LENGTH, registry, run_checks
from reviewbot.diff import collect_diff, parse_diff
from reviewbot.report import format_json, format_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reviewbot",
        description="Review a git diff and report findings.",
    )
    parser.add_argument("--staged", action="store_true",
                        help="Review staged changes (git diff --cached).")
    parser.add_argument("--rev", default=None,
                        help="Review changes since a given revision (e.g. HEAD~1).")
    parser.add_argument("--format", choices=["text", "json"], default="text",
                        help="Output format (default: text).")
    parser.add_argument("--check", action="append", default=None, metavar="NAME",
                        help="Run only this check. Repeat for several. "
                             "See --list-checks for the names.")
    parser.add_argument("--ignore", action="append", default=None, metavar="NAME",
                        help="Skip this check. Repeat for several.")
    parser.add_argument("--max-line-length", type=int, default=DEFAULT_MAX_LINE_LENGTH,
                        help=f"Limit for the long-line check "
                             f"(default: {DEFAULT_MAX_LINE_LENGTH}).")
    parser.add_argument("--list-checks", action="store_true",
                        help="Print every check with its severity and languages, then exit.")
    parser.add_argument("--repo", default=".",
                        help="Path to the git repository (default: current directory).")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _print_checks() -> None:
    print(f"{'CHECK':<22} {'SEVERITY':<9} {'LANGUAGES':<28} DESCRIPTION")
    for name, check in sorted(registry().items()):
        langs = ", ".join(sorted(check.languages)) if check.languages else "all files"
        if len(langs) > 27:
            langs = langs[:24] + "..."
        print(f"{name:<22} {check.severity.value:<9} {langs:<28} {check.description}")


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns an exit code: 0 ok, 1 findings of ERROR severity,
    2 the tool could not run (bad arguments, git failed)."""
    args = parse_args(argv)

    if args.list_checks:
        _print_checks()
        return 0

    repo_path = Path(args.repo).resolve()
    try:
        diff_text = collect_diff(rev=args.rev, staged=args.staged, cwd=repo_path)
    except Exception as e:      # git missing, not a repo, bad revision
        print(f"reviewbot: could not read the diff: {e}", file=sys.stderr)
        return 2

    diffs = parse_diff(diff_text)

    try:
        review = run_checks(
            diffs,
            enabled=set(args.check) if args.check else None,
            ignored=set(args.ignore) if args.ignore else None,
            max_line_length=args.max_line_length,
        )
    except ValueError as e:
        # A mistyped check name must not read as a clean review.
        print(f"reviewbot: {e}", file=sys.stderr)
        return 2

    print(format_json(review) if args.format == "json" else format_text(review))
    return 1 if review.has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
