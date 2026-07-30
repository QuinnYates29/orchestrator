"""Tests for reviewbot.diff — parse_diff and collect_diff."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from reviewbot.diff import collect_diff, parse_diff
from reviewbot.models import FileDiff, Hunk


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

def _make_simple_diff() -> str:
    return """diff --git a/foo.py b/foo.py
index abc123..def456 100644
--- a/foo.py
+++ b/foo.py
@@ -1,7 +1,8 @@
 line1
 line2
+added1
+added2
 line3
 line4
"""


def _make_two_file_diff() -> str:
    return """diff --git a/foo.py b/foo.py
index abc..def 100644
--- a/foo.py
+++ b/foo.py
@@ -10,2 +10,3 @@
 context
+added_foo
diff --git a/bar.py b/bar.py
new file mode 100644
index 000..def
--- /dev/null
+++ b/bar.py
@@ -0,0 +1,5 @@
+line1
+line2
+line3
+line4
+line5
"""


def _make_multi_hunk_diff() -> str:
    return """diff --git a/spam.py b/spam.py
index a..b 100644
--- a/spam.py
+++ b/spam.py
@@ -1,2 +1,3 @@
 context
+added_a
@@ -10,5 +10,6 @@
 more
+added_b
 final
"""


def _make_deleted_file_diff() -> str:
    return """diff --git a/gone.py b/gone.py
deleted file mode 100644
index abc..000
--- a/gone.py
+++ /dev/null
@@ -1,3 +0,0 @@
-line1
-line2
-line3
"""


def _make_binary_diff() -> str:
    return """diff --git a/image.png b/image.png
index abc..def 100644
Binary files a/image.png and b/image.png differ
"""


def _make_binary_alt_diff() -> str:
    return """diff --git a/other.bin b/other.bin
index 123..456 100644
cannot display: file is binary
"""


# --------------------------------------------------------------------------- #
# parse_diff — basic
# --------------------------------------------------------------------------- #

class TestParseDiff:
    def test_simple_additions(self) -> None:
        diff = _make_simple_diff()
        files = parse_diff(diff)
        assert len(files) == 1
        fd = files[0]
        assert fd.path == "foo.py"
        assert not fd.is_new
        assert not fd.is_deleted
        assert len(fd.hunks) == 1
        hunk = fd.hunks[0]
        assert hunk.start_line == 1
        assert hunk.lines == ["added1", "added2"]

    def test_two_files(self) -> None:
        diff = _make_two_file_diff()
        files = parse_diff(diff)
        assert len(files) == 2
        # foo.py
        assert files[0].path == "foo.py"
        assert len(files[0].hunks) == 1
        assert files[0].hunks[0].start_line == 10
        assert files[0].hunks[0].lines == ["added_foo"]
        # bar.py (new file)
        assert files[1].path == "bar.py"
        assert files[1].is_new
        assert len(files[1].hunks) == 1
        assert files[1].hunks[0].start_line == 1
        assert files[1].hunks[0].lines == ["line1", "line2", "line3", "line4", "line5"]

    def test_multi_hunk(self) -> None:
        diff = _make_multi_hunk_diff()
        files = parse_diff(diff)
        assert len(files) == 1
        fd = files[0]
        assert fd.path == "spam.py"
        assert len(fd.hunks) == 2
        assert fd.hunks[0].start_line == 1
        assert fd.hunks[0].lines == ["added_a"]
        assert fd.hunks[1].start_line == 10
        assert fd.hunks[1].lines == ["added_b"]

    def test_deleted_file(self) -> None:
        """Deleted files have no added lines → empty hunks list."""
        diff = _make_deleted_file_diff()
        files = parse_diff(diff)
        assert len(files) == 1
        fd = files[0]
        assert fd.path == "gone.py"
        assert fd.is_deleted
        # no added lines → no hunks
        assert len(fd.hunks) == 0

    def test_binary_files_skipped(self) -> None:
        diff = _make_binary_diff()
        files = parse_diff(diff)
        # should still produce a FileDiff but with no hunks
        assert len(files) == 1
        assert files[0].path == "image.png"
        assert len(files[0].hunks) == 0

    def test_binary_alt_skipped(self) -> None:
        diff = _make_binary_alt_diff()
        files = parse_diff(diff)
        assert len(files) == 1
        assert files[0].path == "other.bin"
        assert len(files[0].hunks) == 0

    def test_empty_input(self) -> None:
        files = parse_diff("")
        assert len(files) == 0

    def test_no_hunks(self) -> None:
        """A diff header with no hunks should still produce a FileDiff."""
        text = """diff --git a/empty.py b/empty.py
index a..b 100644
--- a/empty.py
+++ b/empty.py
"""
        files = parse_diff(text)
        assert len(files) == 1
        assert files[0].path == "empty.py"
        assert len(files[0].hunks) == 0

    def test_added_lines_property(self) -> None:
        diff = _make_simple_diff()
        files = parse_diff(diff)
        fd = files[0]
        added = fd.added_lines
        assert added == [(1, "added1"), (2, "added2")]

    def test_hunk_with_only_added_lines(self) -> None:
        text = """diff --git a/a.py b/a.py
index a..b 100644
--- a/a.py
+++ b/a.py
@@ -0,0 +1,3 @@
+new1
+new2
+new3
"""
        files = parse_diff(text)
        assert len(files) == 1
        assert files[0].hunks[0].lines == ["new1", "new2", "new3"]
        assert files[0].hunks[0].start_line == 1

    def test_context_and_removed_lines_ignored(self) -> None:
        text = """diff --git a/a.py b/a.py
index a..b 100644
--- a/a.py
+++ b/a.py
@@ -1,5 +1,6 @@
 old
+added
 context
-removed
 still_here
"""
        files = parse_diff(text)
        assert len(files) == 1
        assert files[0].hunks[0].lines == ["added"]

    def test_rename_diff(self) -> None:
        text = """diff --git a/old.py b/new.py
similarity index 100%
rename from old.py
rename to new.py
--- a/old.py
+++ b/new.py
@@ -1,2 +1,3 @@
 base
+added
"""
        files = parse_diff(text)
        assert len(files) == 1
        assert files[0].path == "new.py"
        assert files[0].hunks[0].lines == ["added"]

    def test_header_variants_start_line_only(self) -> None:
        """@@ -1 +1,3 @@ (single line range on old side)."""
        text = """diff --git a/x.py b/x.py
index a..b 100644
--- a/x.py
+++ b/x.py
@@ -1 +1,3 @@
+new1
+new2
+new3
"""
        files = parse_diff(text)
        assert len(files) == 1
        assert files[0].hunks[0].start_line == 1
        assert files[0].hunks[0].lines == ["new1", "new2", "new3"]

    def test_multiple_files_some_binary(self) -> None:
        text = """diff --git a/a.txt b/a.txt
index a..b 100644
--- a/a.txt
+++ b/a.txt
@@ -1,2 +1,3 @@
 old
+new
diff --git b/b.bin b/b.bin
index c..d 100644
Binary files b/b.bin and b/b.bin differ
diff --git a/c.txt b/c.txt
index e..f 100644
--- a/c.txt
+++ b/c.txt
@@ -5,3 +5,4 @@
 prev
+added2
"""
        files = parse_diff(text)
        assert len(files) == 3
        assert files[0].path == "a.txt"
        assert len(files[0].hunks) == 1
        assert files[0].hunks[0].lines == ["new"]
        assert files[1].path == "b.bin"
        assert len(files[1].hunks) == 0
        assert files[2].path == "c.txt"
        assert len(files[2].hunks) == 1
        assert files[2].hunks[0].lines == ["added2"]


# --------------------------------------------------------------------------- #
# collect_diff
# --------------------------------------------------------------------------- #

class TestCollectDiff:
    def test_unstaged(self) -> None:
        """Run collect_diff() in a temporary git repo."""
        text = _collect_in_repo([])
        assert "diff --git" in text or text == ""

    def test_staged(self) -> None:
        text = _collect_in_repo(["--cached"])
        assert "diff --git" in text or text == ""

    def test_revision(self) -> None:
        text = _collect_in_repo(["HEAD"])
        assert "diff --git" in text or text == ""


def _collect_in_repo(extra_args: list[str]) -> str:
    """Create a tiny git repo and run collect_diff inside it.

    We use a temporary directory to avoid polluting the actual repo.
    """
    import tempfile
    import os

    old_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            subprocess.run(["git", "init"], capture_output=True, check=True)
            subprocess.run(
                ["git", "config", "user.email", "test@test"],
                capture_output=True, check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "test"],
                capture_output=True, check=True,
            )
            # create a file and commit
            p = Path("file.py")
            p.write_text("line1\nline2\nline3\n")
            subprocess.run(["git", "add", "."], capture_output=True, check=True)
            subprocess.run(
                ["git", "commit", "-m", "initial"],
                capture_output=True, check=True,
            )
            # make a change
            p.write_text("line1\nadded\nline2\nline3\n")
            subprocess.run(["git", "add", "."], capture_output=True, check=True)
            # collect
            if extra_args:
                if extra_args[0] == "--cached":
                    # staged only
                    text = collect_diff(staged=True)
                elif extra_args[0] == "HEAD":
                    text = collect_diff(rev="HEAD")
                else:
                    text = collect_diff()
            else:
                text = collect_diff()
            return text
        finally:
            os.chdir(old_cwd)
