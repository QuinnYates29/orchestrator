"""The checks reviewbot runs over the added lines of a diff.

Two kinds:

* **universal** - run on every file whatever it is, because a merge marker or a
  leaked credential is a problem in YAML and Markdown just as much as in code.
* **language-scoped** - declare the languages they understand and are skipped
  elsewhere, because `except:` means nothing in Go.

Every check matches against `Line.code` (strings and comments blanked) or
`Line.comment`, never the raw text - see languages.split_code_and_comment. The
one deliberate exception is `hardcoded-secret`, which has to read inside string
literals since that is where a secret lives.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable

from reviewbot.languages import Language, Line, build_lines, detect
from reviewbot.models import FileDiff, Finding, Review, Severity

DEFAULT_MAX_LINE_LENGTH = 120


@dataclass
class FileContext:
    """Everything a check needs about one file in the diff."""
    path: str
    language: Language
    lines: list[Line]
    max_line_length: int = DEFAULT_MAX_LINE_LENGTH

    @classmethod
    def from_diff(cls, diff: FileDiff,
                  max_line_length: int = DEFAULT_MAX_LINE_LENGTH) -> "FileContext":
        lang = detect(diff.path)
        return cls(path=diff.path, language=lang,
                   lines=build_lines(diff.added_lines, lang),
                   max_line_length=max_line_length)


CheckFn = Callable[[FileContext], Iterable[Finding]]


@dataclass(frozen=True)
class Check:
    name: str
    severity: Severity
    description: str
    fn: CheckFn
    languages: frozenset[str] = frozenset()   # empty == every language

    def applies_to(self, language: Language) -> bool:
        return not self.languages or language.name in self.languages


_REGISTRY: dict[str, Check] = {}


def _register(name: str, severity: Severity, description: str,
              languages: Iterable[str] = ()) -> Callable[[CheckFn], CheckFn]:
    def wrap(fn: CheckFn) -> CheckFn:
        _REGISTRY[name] = Check(name=name, severity=severity, description=description,
                                fn=fn, languages=frozenset(languages))
        return fn
    return wrap


def _finding(ctx: FileContext, line: Line, name: str, message: str) -> Finding:
    return Finding(path=ctx.path, line=line.number,
                   severity=_REGISTRY[name].severity, check=name, message=message)


# ---------------------------------------------------------------------------
# Universal checks
# ---------------------------------------------------------------------------

_CONFLICT_RE = re.compile(r"^(<{7}|={7}|>{7})(\s|$)")


@_register("merge-conflict", Severity.ERROR,
           "Unresolved merge conflict markers left in a file")
def merge_conflict(ctx: FileContext) -> Iterable[Finding]:
    for line in ctx.lines:
        if _CONFLICT_RE.match(line.raw):
            yield _finding(ctx, line, "merge-conflict",
                           f"unresolved merge conflict marker: {line.raw.strip()[:12]}")


_TODO_RE = re.compile(r"\b(TODO|FIXME|XXX|HACK)\b")


@_register("todo-comment", Severity.INFO, "TODO/FIXME/XXX/HACK left in a comment")
def todo_comment(ctx: FileContext) -> Iterable[Finding]:
    has_comments = bool(ctx.language.line_comments or ctx.language.block_comment)
    for line in ctx.lines:
        # Markdown and plain text have no comment syntax, so for them the whole
        # line is prose and a TODO in it is exactly what this check is for.
        haystack = line.comment if has_comments else line.raw
        match = _TODO_RE.search(haystack)
        if match:
            yield _finding(ctx, line, "todo-comment", f"comment contains {match.group(1)}")


@_register("long-line", Severity.WARNING, "Line longer than --max-line-length")
def long_line(ctx: FileContext) -> Iterable[Finding]:
    for line in ctx.lines:
        length = len(line.raw.rstrip("\n"))
        if length > ctx.max_line_length:
            yield _finding(ctx, line, "long-line",
                           f"line is {length} characters (limit {ctx.max_line_length})")


@_register("trailing-whitespace", Severity.INFO, "Trailing whitespace at end of line")
def trailing_whitespace(ctx: FileContext) -> Iterable[Finding]:
    for line in ctx.lines:
        text = line.raw.rstrip("\n\r")
        if text and text != text.rstrip():
            yield _finding(ctx, line, "trailing-whitespace",
                           f"{len(text) - len(text.rstrip())} trailing whitespace character(s)")


# A secret is *inside* a string literal, so unlike every other check this one
# reads the raw line. Two families: a recognisable key format (few false
# positives worth worrying about), and a secret-sounding name assigned a
# literal (needs the placeholder filter below to stay usable).
_SECRET_PATTERNS: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AWS access key id"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"), "GitHub token"),
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}"), "Slack token"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key"),
    (re.compile(r"\bsk-[A-Za-z0-9]{32,}"), "API secret key"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\."), "JWT"),
)

_ASSIGNED_SECRET_RE = re.compile(
    r"""(?ix)
    \b(?P<name>pass(?:word|wd)?|secret|api[_-]?key|apikey|auth[_-]?token
       |access[_-]?token|private[_-]?key|client[_-]?secret|credentials?)
    \s*[:=]\s*
    (?P<quote>['"])(?P<value>[^'"]{8,})(?P=quote)
    """
)

# Values that look like secrets but are not: placeholders, examples, and
# indirection through the environment or a secret store. Without this the check
# fires on every config template and is turned off within a day.
_NOT_A_SECRET_RE = re.compile(
    r"(?i)^(\s*|x+|\*+|\.+|-+|none|null|true|false|changeme|password|secret|todo"
    r"|your[_-].*|example.*|dummy.*|placeholder.*|test.*|fake.*|sample.*"
    r"|\$\{?[a-z_]+\}?|<[^>]+>|\{\{.*\}\}|%\(.*\)s|\{[a-z_]*\})$"
)
_INDIRECTION_RE = re.compile(
    r"(?i)(os\.environ|getenv|environ\[|process\.env|System\.getenv|ENV\[|"
    r"secretsmanager|vault|keyring|config\.get|settings\.)"
)


@_register("hardcoded-secret", Severity.ERROR,
           "Credential or key committed as a literal")
def hardcoded_secret(ctx: FileContext) -> Iterable[Finding]:
    for line in ctx.lines:
        for pattern, label in _SECRET_PATTERNS:
            if pattern.search(line.raw):
                yield _finding(ctx, line, "hardcoded-secret",
                               f"looks like a committed {label}")
                break
        else:
            if _INDIRECTION_RE.search(line.raw):
                continue   # reading it from the environment is the correct pattern
            match = _ASSIGNED_SECRET_RE.search(line.raw)
            if match and not _NOT_A_SECRET_RE.match(match.group("value")):
                yield _finding(ctx, line, "hardcoded-secret",
                               f"{match.group('name')} assigned a literal value")


# ---------------------------------------------------------------------------
# Language-scoped checks
# ---------------------------------------------------------------------------

# Word-boundary anchored so `pprint(` is its own pattern rather than a stray
# `print(` hit - the original substring match reported `pprint` as `print`, and
# matched inside string literals and comments as well.
_DEBUG_PATTERNS: dict[str, tuple[tuple[re.Pattern, str], ...]] = {
    "python": (
        (re.compile(r"(?<![\w.])print\s*\("), "print()"),
        (re.compile(r"(?<![\w.])pprint\s*\("), "pprint()"),
        (re.compile(r"(?<![\w.])breakpoint\s*\("), "breakpoint()"),
        (re.compile(r"\bi?pdb\.set_trace\s*\("), "pdb.set_trace()"),
    ),
    "javascript": (
        (re.compile(r"\bconsole\.(?:log|debug|dir|trace|info)\s*\("), "console call"),
        (re.compile(r"(?<![\w.])debugger\b"), "debugger statement"),
    ),
    "go": (
        (re.compile(r"\bfmt\.Print(?:ln|f)?\s*\("), "fmt.Print"),
        (re.compile(r"(?<![\w.])println\s*\("), "println()"),
    ),
    "rust": (
        (re.compile(r"(?<![\w.])dbg!\s*\("), "dbg!()"),
        (re.compile(r"(?<![\w.])println!\s*\("), "println!()"),
    ),
    "ruby": (
        (re.compile(r"(?<![\w.])puts\b"), "puts"),
        (re.compile(r"\bbinding\.pry\b"), "binding.pry"),
    ),
    "java": ((re.compile(r"\bSystem\.(?:out|err)\.print"), "System.out.print"),),
    "php": ((re.compile(r"(?<![\w.])(?:var_dump|print_r)\s*\("), "var_dump/print_r"),),
    "c": ((re.compile(r"(?<![\w.])printf\s*\("), "printf()"),),
}
_DEBUG_PATTERNS["typescript"] = _DEBUG_PATTERNS["javascript"]
_DEBUG_PATTERNS["cpp"] = _DEBUG_PATTERNS["c"]


@_register("debug-statement", Severity.WARNING,
           "Debug print or breakpoint left in the code",
           languages=tuple(_DEBUG_PATTERNS))
def debug_statement(ctx: FileContext) -> Iterable[Finding]:
    patterns = _DEBUG_PATTERNS.get(ctx.language.name, ())
    for line in ctx.lines:
        for pattern, label in patterns:
            if pattern.search(line.code):
                yield _finding(ctx, line, "debug-statement", f"{label} left in the code")
                break


# Tolerant of spacing and a trailing comment, which the original exact
# `text.strip() == "except:"` comparison silently let through.
_BARE_EXCEPT_RE = re.compile(r"^\s*except\s*:\s*$")


@_register("bare-except", Severity.ERROR,
           "`except:` also catches SystemExit and KeyboardInterrupt",
           languages=("python",))
def bare_except(ctx: FileContext) -> Iterable[Finding]:
    for line in ctx.lines:
        if _BARE_EXCEPT_RE.match(line.code.rstrip()):
            yield _finding(ctx, line, "bare-except",
                           "bare `except:` - catch a specific exception, or `except Exception:`")


_MUTABLE_DEFAULT_RE = re.compile(r"def\s+\w+\s*\(.*?=\s*(?:\[\s*\]|\{\s*\}|set\s*\(\s*\))")


@_register("mutable-default", Severity.WARNING,
           "Mutable default argument is shared across every call",
           languages=("python",))
def mutable_default(ctx: FileContext) -> Iterable[Finding]:
    for line in ctx.lines:
        if _MUTABLE_DEFAULT_RE.search(line.code):
            yield _finding(ctx, line, "mutable-default",
                           "mutable default argument - use None and build it in the body")


_EMPTY_CATCH_RE = re.compile(r"catch\s*(?:\([^)]*\))?\s*\{\s*\}")


@_register("empty-catch", Severity.WARNING,
           "Exception caught and silently discarded",
           languages=("javascript", "typescript", "java", "csharp", "php"))
def empty_catch(ctx: FileContext) -> Iterable[Finding]:
    for line in ctx.lines:
        if _EMPTY_CATCH_RE.search(line.code):
            yield _finding(ctx, line, "empty-catch",
                           "empty catch block swallows the error")


# ---------------------------------------------------------------------------
# Registry API
# ---------------------------------------------------------------------------

ALL_CHECKS: frozenset[str] = frozenset(_REGISTRY)


def registry() -> dict[str, Check]:
    """Every known check, keyed by name. Used for --list-checks."""
    return dict(_REGISTRY)


def resolve(enabled: Iterable[str] | None = None,
            ignored: Iterable[str] | None = None) -> list[Check]:
    """Which checks to run, after --check and --ignore.

    An unknown name raises rather than being quietly dropped: a typo in
    `--check bare-excepts` that ran nothing would report a clean review, and a
    review tool confidently reporting "no findings" for the wrong reason is the
    most dangerous thing it can do.
    """
    names = set(enabled) if enabled else set(_REGISTRY)
    unknown = sorted(names - set(_REGISTRY))
    if unknown:
        raise ValueError(f"unknown check(s): {', '.join(unknown)} "
                         f"(known: {', '.join(sorted(_REGISTRY))})")
    if ignored:
        unknown_ignored = sorted(set(ignored) - set(_REGISTRY))
        if unknown_ignored:
            raise ValueError(f"unknown check(s) in --ignore: {', '.join(unknown_ignored)} "
                             f"(known: {', '.join(sorted(_REGISTRY))})")
        names -= set(ignored)
    return [_REGISTRY[n] for n in sorted(names)]


def run_checks(diffs: list[FileDiff], enabled: Iterable[str] | None = None,
               ignored: Iterable[str] | None = None,
               max_line_length: int = DEFAULT_MAX_LINE_LENGTH) -> Review:
    """Run the selected checks over every file in `diffs`."""
    checks = resolve(enabled, ignored)
    review = Review()
    reviewed = 0

    for diff in diffs:
        if diff.is_deleted or not diff.added_lines:
            continue    # nothing added means nothing to review
        ctx = FileContext.from_diff(diff, max_line_length=max_line_length)
        reviewed += 1
        for check in checks:
            if check.applies_to(ctx.language):
                review.findings.extend(check.fn(ctx))

    # Previously len(diffs), which counted files the checks skipped - a diff
    # touching one .py and one .md reported "across 2 file(s)" having reviewed
    # exactly one. Count what was actually looked at.
    review.files_reviewed = reviewed
    review.findings.sort(key=lambda f: (f.path, f.line, f.check))
    return review
