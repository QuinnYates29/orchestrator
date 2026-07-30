# reviewbot

A small command-line tool that reviews a git diff and reports findings.

    reviewbot                      # review unstaged changes
    reviewbot --staged             # review staged changes
    reviewbot --rev HEAD~1         # review changes since a revision
    reviewbot --format json        # machine-readable output

## Conventions

- Python 3.11+, standard library only. No third-party runtime dependencies.
- `from __future__ import annotations` at the top of every module.
- Every module has a matching `tests/test_<module>.py`, written with plain
  `pytest` functions - no classes, no fixtures beyond `tmp_path`.
- `reviewbot/models.py` is the shared contract. Import from it; do not edit it.
- Type-annotate public functions. Comments explain *why*, not *what*.

## Running the tests

    python3 -m pytest tests/ -q
