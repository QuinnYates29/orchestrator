Build out `reviewbot`, a small command-line tool that reviews a git diff and
reports findings. The repository already contains the shared data model and the
project conventions — read `reviewbot/models.py` and `README.md` first.

The work splits into four pieces. Each one owns its files completely, and no
two pieces touch the same file.

Pieces 1, 2 and 3 are fully independent of each other and must run in parallel.
Piece 4 imports all three, so it **must declare a dependency on pieces 1, 2 and
3** and run after they are integrated — it cannot be written against modules
that do not exist yet.

## 1. Diff parsing — `reviewbot/diff.py` + `tests/test_diff.py`

Parse unified diff text into the `FileDiff` / `Hunk` types from
`reviewbot.models`.

- `parse_diff(text: str) -> list[FileDiff]` — parse `git diff` output.
  Handle multiple files, multiple hunks per file, new files (`--- /dev/null`),
  and deleted files. Record only added lines in `Hunk.lines`, with the `+`
  stripped, and set `Hunk.start_line` to the correct 1-indexed line number in
  the new file (from the `@@ -a,b +c,d @@` header).
- `collect_diff(rev: str | None = None, staged: bool = False, cwd: Path | None = None) -> str`
  — run the right `git diff` command via `subprocess` and return its stdout.
  Use `git diff` for unstaged, `git diff --cached` for staged, and
  `git diff <rev>` when a revision is given.

Deleted files and binary files must not crash the parser.

## 2. Checks — `reviewbot/checks.py` + `tests/test_checks.py`

Each check looks at the added lines of a `FileDiff` and returns `Finding`
objects. Implement exactly these four, and only for files ending in `.py`:

- `debug-print` (WARNING): a line containing a bare `print(` call.
- `todo-comment` (INFO): a line containing `TODO` or `FIXME`.
- `bare-except` (ERROR): a line whose stripped form is `except:`.
- `long-line` (WARNING): a line longer than 100 characters.

Expose `ALL_CHECKS: list[Check]` where each `Check` is a callable taking a
`FileDiff` and returning `list[Finding]`, plus:

- `run_checks(diffs: list[FileDiff], enabled: set[str] | None = None) -> Review`
  — run every check (or only those whose names are in `enabled`) over every
  file and return a populated `Review` with `files_reviewed` set.

Set each `Finding.line` to the real line number in the new file.

## 3. Reporting — `reviewbot/report.py` + `tests/test_report.py`

Turn a `Review` into output.

- `format_text(review: Review) -> str` — one finding per line, in the form
  `path:line: SEVERITY [check] message`, grouped by file with files in sorted
  order and findings within a file sorted by line number. End with a summary
  line: `N finding(s) across M file(s)`. A `Review` with no findings must
  produce `No findings.` and nothing else.
- `format_json(review: Review) -> str` — a JSON object with keys `findings`
  (a list of objects with `path`, `line`, `severity`, `check`, `message`) and
  `files_reviewed`. Indent by 2. `severity` is the string value, not the enum.

## 4. Command line — `reviewbot/cli.py` + `tests/test_cli.py`

Wire the three pieces together with `argparse`.

- Flags: `--staged`, `--rev REV`, `--format {text,json}` (default `text`),
  `--check NAME` (repeatable; limits which checks run), and
  `--repo PATH` (default: current directory).
- `main(argv: list[str] | None = None) -> int` — collect the diff, parse it,
  run the checks, print the formatted report to stdout, and return an exit
  code: `1` if the review has any ERROR finding, otherwise `0`.
- A `if __name__ == "__main__":` block calling `sys.exit(main())`.

This piece runs after 1-3 are integrated, so import them normally
(`from reviewbot.diff import parse_diff, collect_diff`). Test `main()` by
monkeypatching `collect_diff` to return diff text of your own — do not
reimplement parsing, checking or formatting here, and do not shell out to git
in a test.

## Requirements for every piece

- Standard library only.
- Write your tests as you go and run them: `python3 -m pytest tests/ -q`.
  A piece is not done until its own tests pass.
- Do not edit `reviewbot/models.py`, `README.md`, or any file belonging to
  another piece.
