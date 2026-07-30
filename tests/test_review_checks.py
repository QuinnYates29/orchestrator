"""Checks: what they catch, and - more importantly - what they don't."""
from __future__ import annotations

import pytest

from pipeline.review.checks import ALL_CHECKS, registry, resolve, run_checks
from pipeline.review.models import FileDiff, Hunk, Severity


def _diff(path: str, *lines: str, start: int = 1) -> FileDiff:
    return FileDiff(path=path, hunks=[Hunk(start_line=start, lines=list(lines))])


def _run(path: str, *lines: str, start: int = 1, **kw) -> list:
    return run_checks([_diff(path, *lines, start=start)], **kw).findings


def _checks(path: str, *lines: str, **kw) -> set[str]:
    return {f.check for f in _run(path, *lines, **kw)}


# ---------------------------------------------------------------------------
# Precision: the false positives that made the first version noisy
# ---------------------------------------------------------------------------

def test_print_inside_a_string_is_not_a_debug_statement():
    """The original substring match flagged this. It is a string literal."""
    assert "debug-statement" not in _checks("a.py", 'msg = "use print(x) to debug"')


def test_print_inside_a_comment_is_not_a_debug_statement():
    assert "debug-statement" not in _checks("a.py", "# remember to print(x) here")


def test_pprint_is_reported_as_pprint_not_as_print():
    """`pprint(` used to match the `print(` pattern and be mislabelled."""
    findings = _run("a.py", "    pprint(payload)")
    assert [f.check for f in findings] == ["debug-statement"]
    assert "pprint()" in findings[0].message


def test_a_method_named_print_is_not_a_bare_print():
    assert "debug-statement" not in _checks("a.py", "    self.printer.print(x)")
    assert "debug-statement" not in _checks("a.py", "    widget.print()")


def test_a_real_print_is_still_caught():
    assert "debug-statement" in _checks("a.py", "    print(value)")


def test_todo_in_code_is_not_a_todo_comment():
    """A variable called TODO_LIST is not a TODO comment."""
    assert "todo-comment" not in _checks("a.py", 'TODO_LIST = ["a"]')


def test_todo_in_a_comment_is_caught():
    assert "todo-comment" in _checks("a.py", "x = 1  # TODO: fix this")


@pytest.mark.parametrize("word", ["TODO", "FIXME", "XXX", "HACK"])
def test_todo_variants_are_caught(word):
    assert "todo-comment" in _checks("a.py", f"# {word}: something")


def test_markdown_prose_counts_as_a_comment():
    """Markdown has no comment syntax, so the prose is the comment."""
    assert "todo-comment" in _checks("notes.md", "- TODO: write the guide")


# ---------------------------------------------------------------------------
# bare-except, now tolerant of the spacing the exact match missed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("line", [
    "except:",
    "    except:",
    "    except :",
    "    except:  # noqa",
    "    except:   ",
])
def test_bare_except_variants(line):
    assert "bare-except" in _checks("a.py", line)


@pytest.mark.parametrize("line", [
    "    except ValueError:",
    "    except Exception:",
    "    except (A, B):",
])
def test_qualified_except_is_fine(line):
    assert "bare-except" not in _checks("a.py", line)


def test_bare_except_is_an_error():
    (finding,) = _run("a.py", "    except:")
    assert finding.severity is Severity.ERROR


# ---------------------------------------------------------------------------
# Every file type, not just Python
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path,line", [
    ("app.js", "  console.log(user)"),
    ("app.ts", "  debugger;"),
    ("main.go", "\tfmt.Println(err)"),
    ("main.rs", "    dbg!(&value);"),
    ("app.rb", "  binding.pry"),
    ("A.java", "    System.out.println(x);"),
    ("x.php", "  var_dump($x);"),
])
def test_debug_statements_across_languages(path, line):
    assert "debug-statement" in _checks(path, line)


def test_python_checks_do_not_run_on_go():
    """`except:` is not a thing in Go and must not be reported there."""
    assert "bare-except" not in _checks("main.go", "except:")


def test_javascript_checks_do_not_run_on_python():
    assert "debug-statement" not in _checks("a.py", "console.log(x)")


def test_unknown_extension_still_gets_universal_checks():
    """The point of not being Python-only: an unrecognised file is reviewed,
    not skipped."""
    assert "long-line" in _checks("data.weird", "x" * 200)


def test_empty_catch_in_javascript():
    assert "empty-catch" in _checks("a.js", "  try { go() } catch (e) {}")


def test_mutable_default_argument():
    assert "mutable-default" in _checks("a.py", "def f(items=[]):")
    assert "mutable-default" in _checks("a.py", "def f(opts={}):")
    assert "mutable-default" not in _checks("a.py", "def f(items=None):")


# ---------------------------------------------------------------------------
# Universal checks
# ---------------------------------------------------------------------------

def test_merge_conflict_markers_are_an_error():
    findings = _run("a.py", "<<<<<<< HEAD", "x = 1", "=======", "x = 2", ">>>>>>> other")
    conflicts = [f for f in findings if f.check == "merge-conflict"]
    assert len(conflicts) == 3
    assert all(f.severity is Severity.ERROR for f in conflicts)


def test_merge_conflict_found_in_any_file_type():
    assert "merge-conflict" in _checks("config.yaml", "<<<<<<< HEAD")


def test_a_row_of_equals_in_prose_is_not_a_conflict():
    """Underlines and separators are common; the marker is exactly seven."""
    assert "merge-conflict" not in _checks("a.md", "=" * 40)


def test_trailing_whitespace():
    assert "trailing-whitespace" in _checks("a.py", "x = 1   ")
    assert "trailing-whitespace" not in _checks("a.py", "x = 1")


def test_long_line_respects_the_configured_limit():
    line = "x" * 130
    assert "long-line" in _checks("a.py", line)
    assert "long-line" not in _checks("a.py", line, max_line_length=200)
    assert "long-line" in _checks("a.py", "y" * 50, max_line_length=10)


# ---------------------------------------------------------------------------
# hardcoded-secret: worth having only if it is not constantly wrong
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("line", [
    'AWS_KEY = "AKIAIOSFODNN7EXAMPLE"',
    'token = "ghp_012345678901234567890123456789012345"',
    'slack = "xoxb-123456789012-abcdefghij"',
    'key = "-----BEGIN RSA PRIVATE KEY-----"',
    'password = "hunter2istheworst"',
    'api_key: "9f8a7b6c5d4e3f2a1b0c9d8e"',
])
def test_secrets_are_caught(line):
    assert "hardcoded-secret" in _checks("a.py", line)


@pytest.mark.parametrize("line", [
    'password = os.environ["DB_PASSWORD"]',
    'password = os.getenv("DB_PASSWORD")',
    'const secret = process.env.SECRET',
    'password = ""',
    'password = "changeme"',
    'api_key = "your-api-key-here"',
    'password = "xxxxxxxxxx"',
    'api_key = "${API_KEY}"',
    'password = "<your password>"',
    'token = "{{ vault_token }}"',
    'password = "example-value"',
])
def test_placeholders_and_env_lookups_are_not_secrets(line):
    """Without this the check fires on every config template and gets muted."""
    assert "hardcoded-secret" not in _checks("app.py", line)


def test_secret_in_a_yaml_file_is_caught():
    assert "hardcoded-secret" in _checks("config.yaml", 'api_key: "9f8a7b6c5d4e3f2a1b"')


def test_secret_is_an_error():
    (finding,) = _run("a.py", 'password = "hunter2istheworst"')
    assert finding.severity is Severity.ERROR


# ---------------------------------------------------------------------------
# Registry, selection, counting
# ---------------------------------------------------------------------------

def test_every_registered_check_is_in_all_checks():
    assert ALL_CHECKS == frozenset(registry())
    assert len(ALL_CHECKS) >= 8


def test_every_check_has_a_description_and_severity():
    for name, check in registry().items():
        assert check.description, name
        assert isinstance(check.severity, Severity), name


def test_enabled_selects_only_those_checks():
    findings = _run("a.py", "    except:", "    print(x)", enabled={"bare-except"})
    assert {f.check for f in findings} == {"bare-except"}


def test_ignored_removes_a_check():
    names = {f.check for f in _run("a.py", "    except:", "    print(x)",
                                   ignored={"debug-statement"})}
    assert "debug-statement" not in names
    assert "bare-except" in names


def test_an_unknown_check_name_raises_rather_than_reporting_clean():
    """A typo that silently ran nothing would report a clean review - the most
    dangerous answer a review tool can give."""
    with pytest.raises(ValueError, match="unknown check"):
        resolve(enabled={"bare-excepts"})
    with pytest.raises(ValueError, match="unknown check"):
        resolve(ignored={"nope"})


def test_files_reviewed_counts_only_reviewed_files():
    """Used to be len(diffs), so a diff touching a deleted file claimed to have
    reviewed it."""
    review = run_checks([
        _diff("a.py", "print(x)"),
        _diff("notes.md", "hello"),
        FileDiff(path="gone.py", hunks=[], is_deleted=True),
    ])
    assert review.files_reviewed == 2


def test_a_file_with_no_added_lines_is_not_counted():
    review = run_checks([FileDiff(path="empty.py", hunks=[])])
    assert review.files_reviewed == 0


def test_findings_come_back_sorted():
    review = run_checks([
        _diff("b.py", "    except:"),
        _diff("a.py", "x" * 200, "    print(y)"),
    ])
    keys = [(f.path, f.line) for f in review.findings]
    assert keys == sorted(keys)


def test_line_numbers_point_at_the_real_line():
    findings = _run("a.py", "ok = 1", "    except:", start=40)
    assert [f.line for f in findings] == [41]
