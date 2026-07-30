"""Diff parsing and collection utilities.

parse_diff() converts unified git diff text into FileDiff/Hunk objects.
collect_diff() runs git diff to obtain that text.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from reviewbot.models import FileDiff, Hunk

# --------------------------------------------------------------------------- #
# Parse
# --------------------------------------------------------------------------- #

_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")

def parse_diff(text: str) -> list[FileDiff]:
    """Parse unified git diff *text* into a list of FileDiff objects.

    Each ``diff --git`` section produces one ``FileDiff``.  Hunks are
    extracted from ``@@ ... @@`` blocks.  Only added lines (``+``) are kept,
    with the leading ``+`` stripped.  Metadata lines (``---``, ``+++``,
    ``diff --git``, ``index``, ``new/deleted file mode``) are ignored.
    Binary-file markers are detected and silently skipped.
    """
    files: list[FileDiff] = []
    lines = text.splitlines()

    # state
    current_file: FileDiff | None = None
    in_hunk = False
    current_start_line = 0
    current_added: list[str] = []
    saw_binary = False

    for raw in lines:
        line = raw.rstrip("\n\r")

        # ---- detect binary and skip the file ---------------------------------
        if line.startswith("Binary files ") or "cannot display" in line:
            saw_binary = True
            continue

        # ---- new file section ------------------------------------------------
        if line.startswith("diff --git "):
            # flush previous file
            if current_file is not None and not saw_binary:
                if current_added:
                    current_file.hunks.append(
                        Hunk(start_line=current_start_line, lines=list(current_added))
                    )
                files.append(current_file)
            elif current_file is not None and saw_binary:
                files.append(current_file)   # no hunks, that's fine
            # reset
            current_file = None
            in_hunk = False
            current_added.clear()
            saw_binary = False

            # extract path after "diff --git "
            parts = line[len("diff --git "):].split()
            if parts:
                # a/b a/b -> pick the second (or the first if only one)
                path = parts[-1]
                if path.startswith("b/"):
                    path = path[2:]
                current_file = FileDiff(path=path)
            else:
                current_file = FileDiff(path="?")
            continue

        # ---- metadata lines -------------------------------------------------
        if line.startswith("--- ") or line.startswith("+++ ") or line.startswith("index "):
            continue
        if line.startswith("new file mode"):
            if current_file is not None:
                current_file.is_new = True
            continue
        if line.startswith("deleted file mode"):
            if current_file is not None:
                current_file.is_deleted = True
            continue
        if line.startswith("similarity index"):
            continue
        if line.startswith("rename from") or line.startswith("rename to"):
            continue

        # ---- hunk header ----------------------------------------------------
        m = _HUNK_HEADER_RE.match(line)
        if m:
            # flush previous hunk
            if current_file is not None and current_added and not saw_binary:
                current_file.hunks.append(
                    Hunk(start_line=current_start_line, lines=list(current_added))
                )
            current_added.clear()
            # new hunk: start_line is from the '+' side (new file)
            current_start_line = int(m.group(2))
            in_hunk = True
            continue

        # ---- context / added / removed lines inside a hunk -------------------
        if not in_hunk or current_file is None:
            continue
        if saw_binary:
            continue

        if line.startswith("+") and not line.startswith("+++"):
            # added line
            current_added.append(line[1:])
        elif line.startswith("-"):
            # removed line – skip
            pass
        elif line.startswith(" "):
            # context line – skip
            pass

    # flush last file
    if current_file is not None:
        if current_added and not saw_binary:
            current_file.hunks.append(
                Hunk(start_line=current_start_line, lines=list(current_added))
            )
        files.append(current_file)

    return files


# --------------------------------------------------------------------------- #
# Collect
# --------------------------------------------------------------------------- #

def collect_diff(
    *,
    rev: str | None = None,
    staged: bool = False,
    cwd: Path | None = None,
) -> str:
    """Run ``git diff`` and return the diff text.

    Parameters
    ----------
    rev : str, optional
        If given, diff against this revision (``git diff <revision>``).
        When *rev* is set, *staged* is ignored.
    staged : bool
        If True and no *rev* is given, diff staged changes
        (``git diff --cached``).  Default: unstaged changes.
    cwd : Path, optional
        Path to the git repository (default: current directory).

    Returns
    -------
    str
        The raw diff output (UTF-8 decoded).
    """
    cmd = ["git", "diff"]

    if rev:
        cmd.append(rev)
    elif staged:
        cmd.append("--cached")

    if cwd is not None:
        result = subprocess.run(cmd, capture_output=True, check=True, cwd=str(cwd))
    else:
        result = subprocess.run(cmd, capture_output=True, check=True)
    return result.stdout.decode("utf-8", errors="replace")
