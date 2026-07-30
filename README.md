# reviewbot

Reviews a git diff and reports findings. Standard library only, no runtime
dependencies. Works on any file type, not just Python.

It looks at **added lines only** — the lines a diff introduces — so it reports
on what a change brings in rather than re-litigating the whole file.

## Install

```bash
pip install -e .        # then `reviewbot` works from anywhere
```

Or run it without installing, from the repository root:

```bash
python3 -m reviewbot
```

## Usage

```bash
reviewbot                              # unstaged changes
reviewbot --staged                     # staged changes
reviewbot --rev HEAD~1                 # changes since a revision
reviewbot --repo /path/to/project      # a different repository
reviewbot --format json                # machine-readable
reviewbot --check bare-except          # only this check (repeatable)
reviewbot --ignore todo-comment        # everything but this (repeatable)
reviewbot --max-line-length 100        # default is 120
reviewbot --list-checks                # what it can find
```

Exit codes: **0** no errors, **1** at least one ERROR finding, **2** the tool
could not run (not a repo, bad revision, unknown check name).

That makes it usable as a gate:

```bash
reviewbot --staged || exit 1           # .git/hooks/pre-commit
reviewbot --rev origin/main            # CI
```

## Checks

Universal — run on every file:

| Check | Severity | Finds |
|---|---|---|
| `merge-conflict` | error | `<<<<<<<` / `=======` / `>>>>>>>` left in a file |
| `hardcoded-secret` | error | AWS/GitHub/Slack keys, private keys, JWTs, and secret-named variables assigned a literal |
| `long-line` | warning | Lines over `--max-line-length` |
| `todo-comment` | info | `TODO` / `FIXME` / `XXX` / `HACK` in a comment |
| `trailing-whitespace` | info | Whitespace at end of line |

Language-scoped — skipped where they don't apply:

| Check | Severity | Languages |
|---|---|---|
| `debug-statement` | warning | Python, JS/TS, Go, Rust, Ruby, Java, PHP, C/C++ |
| `bare-except` | error | Python |
| `mutable-default` | warning | Python |
| `empty-catch` | warning | JS/TS, Java, C#, PHP |

`--list-checks` prints this table at runtime.

## Precision

Every check runs against a line whose **string contents and comments have been
masked out** first (`reviewbot/languages.py`). That is what keeps it quiet:

```python
msg = "use print(x) to debug"   # not a debug statement - it is a string
# remember to print(x) here     # not a debug statement - it is a comment
self.printer.print(x)           # not a bare print()
pprint(payload)                 # reported as pprint(), not mislabelled print()
except:  # noqa                 # still caught, despite the trailing comment
```

`hardcoded-secret` is the deliberate exception — it reads the raw line, because
a secret lives inside the string literal everything else is ignoring. It skips
placeholders (`changeme`, `your-api-key`, `${VAR}`, `<value>`, `{{ tpl }}`) and
anything read from the environment or a secret store, without which it fires on
every config template and gets muted within a day.

## Known limits

- **Regex, not a parser.** No AST, no type information, no cross-file analysis.
- **Line-by-line masking.** A diff gives added lines without surrounding
  context, so there is no way to know whether a hunk began inside a multi-line
  string. A `"""docstring"""` spanning several added lines is only partly
  masked. Reporting is preferred to staying silent: a false positive costs a
  glance, a missed `except:` costs more.
- **Added lines only.** Problems already in the file are not reported.

## Conventions

- Python 3.11+, standard library only.
- `from __future__ import annotations` at the top of every module.
- Every module has a matching `tests/test_<module>.py`, plain `pytest`
  functions, no fixtures beyond `tmp_path`.
- `reviewbot/models.py` is the shared contract. Import from it; do not edit it.

## Tests

```bash
python3 -m pytest tests/ -q
```
