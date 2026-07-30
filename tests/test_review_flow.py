"""The `pipeline review` workflow.

The load-bearing test here is the read-only one: this workflow must not be able
to modify the repository it is reviewing, and that has to be structural rather
than a promise in a prompt.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from pipeline.events import EventLog
from pipeline.review.flow import (
    SUBMIT_FINDINGS_TOOL,
    FileReview,
    _merge_findings,
    _parse_submitted,
    run_review,
    split_diff_sections,
)
from pipeline.review.models import Finding, Review, Severity


def _run(coro):
    return asyncio.run(coro)


DIFF = """\
diff --git a/app.py b/app.py
index 111..222 100644
--- a/app.py
+++ b/app.py
@@ -1,3 +1,5 @@
 def handler(x):
+    print(x)
+    return x[0]

diff --git a/notes.md b/notes.md
index 333..444 100644
--- a/notes.md
+++ b/notes.md
@@ -1 +1,2 @@
 hello
+TODO: write this up
"""


class FakeClient:
    """Records every request so the tool surface can be asserted on."""

    def __init__(self, findings_by_path=None, summary="Looks fine."):
        self.findings_by_path = findings_by_path or {}
        self.summary = summary
        self.calls = []

    async def chat_once(self, model, messages, *, tools=None, tool_choice=None,
                        max_tokens=None, **kw):
        self.calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
        if tools is None:                      # the summary stage
            return {"choices": [{"message": {"role": "assistant", "content": self.summary}}],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 5}}
        path = _path_under_review(messages)
        return {
            "choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [{
                "id": "c1", "type": "function", "function": {
                    "name": "submit_findings",
                    "arguments": json.dumps({"findings": self.findings_by_path.get(path, [])}),
                }}]}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        }

    async def admin_load(self, model, profile=None, wait_s=180.0):
        return {"ok": True}


def _path_under_review(messages) -> str:
    for m in messages:
        content = m.get("content") or ""
        if "File under review" in content:
            return content.split("`")[1]
    return ""


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setattr("pipeline.review.flow.collect_diff",
                        lambda rev=None, staged=False, cwd=None: DIFF)
    return tmp_path


# ---------------------------------------------------------------------------
# The guarantee: this workflow cannot write
# ---------------------------------------------------------------------------

def test_agents_are_never_offered_a_tool_that_writes(repo):
    """Structural, not a promise in a prompt: write_file, edit_file and
    run_shell are absent from the request entirely."""
    client = FakeClient()
    _run(run_review(client, repo=repo, model="ds4"))

    saw_tools = False
    for call in client.calls:
        for tool in call["tools"] or []:
            saw_tools = True
            assert tool["function"]["name"] in {
                "read_file", "list_dir", "grep", "glob", "submit_findings"
            }, tool["function"]["name"]
    assert saw_tools, "no tools were offered at all - the assertion proved nothing"


def test_review_leaves_the_repository_untouched(repo, monkeypatch):
    (repo / "app.py").write_text("original\n")
    before = {p: p.read_bytes() for p in repo.rglob("*") if p.is_file()}

    _run(run_review(FakeClient(), repo=repo, model="ds4"))

    after = {p: p.read_bytes() for p in repo.rglob("*") if p.is_file()}
    assert after == before


def test_submit_findings_tool_has_no_fix_or_patch_field():
    """The tool shape is what stops a model reporting an edit it 'made'."""
    props = SUBMIT_FINDINGS_TOOL["function"]["parameters"]["properties"]
    item = props["findings"]["items"]["properties"]
    assert set(item) == {"line", "severity", "category", "message"}


# ---------------------------------------------------------------------------
# Static pass
# ---------------------------------------------------------------------------

def test_static_only_makes_no_model_calls(repo):
    client = FakeClient()
    run = _run(run_review(client, repo=repo, model="ds4", static_only=True))
    assert client.calls == []
    assert run.static_findings > 0
    assert run.model_findings == 0


def test_static_pass_covers_every_file_type(repo):
    run = _run(run_review(None, repo=repo, static_only=True))
    paths = {f.path for f in run.review.findings}
    assert "app.py" in paths        # debug-statement
    assert "notes.md" in paths      # todo-comment


# ---------------------------------------------------------------------------
# Model pass
# ---------------------------------------------------------------------------

def test_model_findings_are_merged_and_labelled(repo):
    client = FakeClient(findings_by_path={
        "app.py": [{"line": 3, "severity": "error", "category": "logic-error",
                    "message": "x[0] raises IndexError on an empty list"}],
    })
    run = _run(run_review(client, repo=repo, model="ds4"))

    model_findings = [f for f in run.review.findings if f.source == "model"]
    assert len(model_findings) == 1
    assert model_findings[0].check == "review/logic-error"
    assert model_findings[0].severity is Severity.ERROR
    assert run.model_findings == 1


def test_a_model_finding_on_an_already_flagged_line_is_dropped():
    """The static finding names the exact rule it matched, so it wins."""
    static = Review(findings=[Finding(path="a.py", line=2, severity=Severity.WARNING,
                                      check="debug-statement", message="print()")])
    reviews = [FileReview(path="a.py", findings=[
        Finding(path="a.py", line=2, severity=Severity.INFO, check="review/style",
                message="debug print", source="model"),
        Finding(path="a.py", line=9, severity=Severity.ERROR, check="review/logic",
                message="off by one", source="model"),
    ])]
    merged = _merge_findings(static, reviews)
    assert [(f.line, f.source) for f in merged] == [(2, "static"), (9, "model")]


def test_findings_are_sorted(repo):
    client = FakeClient(findings_by_path={
        "app.py": [{"line": 99, "severity": "info", "message": "later"},
                   {"line": 4, "severity": "info", "message": "earlier"}],
    })
    run = _run(run_review(client, repo=repo, model="ds4"))
    keys = [(f.path, f.line) for f in run.review.findings]
    assert keys == sorted(keys)


def test_summary_is_attached(repo):
    run = _run(run_review(FakeClient(summary="Two small issues."), repo=repo, model="ds4"))
    assert run.review.summary == "Two small issues."


def test_no_summary_skips_that_call(repo):
    client = FakeClient()
    _run(run_review(client, repo=repo, model="ds4", summarise=False))
    assert all(c["tools"] is not None for c in client.calls), "a summary call was still made"


def test_max_files_caps_the_model_pass_but_not_the_static_one(repo):
    client = FakeClient()
    run = _run(run_review(client, repo=repo, model="ds4", max_files=1))
    assert len(run.file_reviews) == 1
    assert run.review.files_reviewed == 2, "the static pass must still see every file"


def test_largest_change_is_reviewed_first_when_capped(repo):
    client = FakeClient()
    run = _run(run_review(client, repo=repo, model="ds4", max_files=1))
    # app.py adds two lines, notes.md adds one.
    assert run.file_reviews[0].path == "app.py"


# ---------------------------------------------------------------------------
# Not reporting a partial review as a clean one
# ---------------------------------------------------------------------------

def test_a_file_that_never_submits_is_reported_as_unreviewed(repo):
    class SilentClient(FakeClient):
        async def chat_once(self, model, messages, *, tools=None, **kw):
            self.calls.append({"messages": messages, "tools": tools, "tool_choice": None})
            if tools is None:
                return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
            return {"choices": [{"message": {"role": "assistant", "content": "still reading"}}]}

    run = _run(run_review(SilentClient(), repo=repo, model="ds4", max_turns=2))
    assert len(run.unreviewed) == 2
    assert all("did not submit" in f.error for f in run.unreviewed)


def test_a_backend_error_on_one_file_does_not_sink_the_review(repo):
    from pipeline.client import OrchestratorError

    class FlakyClient(FakeClient):
        async def chat_once(self, model, messages, *, tools=None, **kw):
            if tools and _path_under_review(messages) == "app.py":
                raise OrchestratorError(503, {"error": "model_not_resident"})
            return await super().chat_once(model, messages, tools=tools, **kw)

    run = _run(run_review(FlakyClient(), repo=repo, model="ds4"))
    assert [f.path for f in run.unreviewed] == ["app.py"]
    assert run.review.findings, "the static findings must survive"


def test_last_turn_withdraws_the_reading_tools(repo):
    """Same trap as the planner: a system prompt telling the model to keep
    reading beats tool_choice, so the final turn must remove both."""
    class SilentClient(FakeClient):
        async def chat_once(self, model, messages, *, tools=None, tool_choice=None, **kw):
            self.calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
            if tools is None:
                return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
            return {"choices": [{"message": {"role": "assistant", "content": "reading"}}]}

    client = SilentClient()
    _run(run_review(client, repo=repo, model="ds4", max_turns=3, max_files=1))
    final = [c for c in client.calls if c["tools"]][-1]
    assert [t["function"]["name"] for t in final["tools"]] == ["submit_findings"]
    assert final["tool_choice"]["function"]["name"] == "submit_findings"
    assert "no other tools" in final["messages"][0]["content"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_split_diff_sections_keeps_removed_lines_and_context():
    """parse_diff keeps only added lines, which is not enough to review by."""
    sections = split_diff_sections(DIFF)
    assert set(sections) == {"app.py", "notes.md"}
    assert "def handler(x):" in sections["app.py"]
    assert "@@" in sections["app.py"]


def test_unknown_severity_becomes_a_warning_rather_than_being_dropped():
    (finding,) = _parse_submitted("a.py", json.dumps(
        {"findings": [{"line": 1, "severity": "critical", "message": "m"}]}))
    assert finding.severity is Severity.WARNING


def test_a_finding_with_no_message_is_discarded():
    assert _parse_submitted("a.py", json.dumps(
        {"findings": [{"line": 1, "severity": "error"}]})) == []


def test_malformed_submission_does_not_raise():
    assert _parse_submitted("a.py", json.dumps({"findings": "not a list"})) == []
    assert _parse_submitted("a.py", json.dumps({})) == []


def test_events_record_the_run(repo, tmp_path):
    events = EventLog(tmp_path / "e.jsonl", run_id="r")
    _run(run_review(FakeClient(), repo=repo, model="ds4", events=events))
    kinds = {e["kind"] for e in EventLog.read(events.path)}
    assert {"review_static_done", "review_agents_start", "review_end"} <= kinds
    usage = [e for e in EventLog.read(events.path) if e["kind"] == "usage"]
    assert usage and all(e["reported"] for e in usage)
