#!/usr/bin/env python3
"""Live view of a pipeline run's event log.

    ./watch-run.py /path/to/repo          # follow the newest run
    ./watch-run.py /path/to/repo <run_id> # follow a specific one
    ./watch-run.py /path/to/repo --once   # print what has happened and exit

The run's own stderr is a log stream meant for diagnosis after the fact. This
is the at-a-glance view: what each agent is doing right now, what it costs, and
which things have already gone wrong. It tails the JSONL event log, so it is
read-only and safe to start, stop, and restart mid-run.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

C = {
    "dim": "\033[2m", "red": "\033[31m", "green": "\033[32m", "yellow": "\033[33m",
    "blue": "\033[34m", "magenta": "\033[35m", "cyan": "\033[36m", "bold": "\033[1m",
    "off": "\033[0m",
}
if not sys.stdout.isatty():
    C = {k: "" for k in C}


def _newest_run(scratch: Path) -> Path | None:
    runs = [d for d in scratch.iterdir() if d.is_dir() and (d / "events.jsonl").exists()]
    return max(runs, key=lambda d: d.stat().st_mtime) if runs else None


def _clip(text: str, width: int = 90) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 1] + "…"


def _render(e: dict, totals: dict) -> str | None:
    """One line for one event, or None for events not worth a line.

    Usage events are the noisiest thing in the log and the least useful
    individually, so they are accumulated into a running total and reported
    against the events that mean something instead.
    """
    kind = e.get("kind", "")
    ts = time.strftime("%H:%M:%S", time.localtime(e.get("ts", 0)))
    stamp = f"{C['dim']}{ts}{C['off']}"

    if kind == "usage":
        totals["prompt"] += e.get("prompt_tokens", 0)
        totals["completion"] += e.get("completion_tokens", 0)
        totals["calls"] += 1
        if not e.get("reported", True):
            totals["unreported"] += 1
        return None

    def cost() -> str:
        if not totals["calls"]:
            return ""
        flag = "~" if totals["unreported"] else ""
        return (f" {C['dim']}[{flag}{totals['prompt'] // 1000}k in / "
                f"{totals['completion'] // 1000}k out, {totals['calls']} calls]{C['off']}")

    if kind == "run_start":
        roles = e.get("roles") or {}
        return (f"{stamp} {C['bold']}RUN{C['off']} {_clip(e.get('task', ''), 70)}\n"
                f"{stamp} {C['dim']}roles: "
                f"{', '.join(f'{k}={v}' for k, v in roles.items())}{C['off']}")
    if kind == "plan":
        chunks = e.get("chunks") or []
        head = f"{stamp} {C['bold']}PLAN{C['off']} {len(chunks)} chunk(s){cost()}"
        return "\n".join([head] + [f"{stamp}   {C['cyan']}{c['id']}{C['off']} {_clip(c['title'], 70)}"
                                   for c in chunks])
    if kind == "plan_rejected":
        return f"{stamp} {C['yellow']}PLAN REJECTED{C['off']} {_clip(e.get('reason', ''))}"
    if kind == "chunk_started":
        return f"{stamp} {C['blue']}START{C['off']} {e.get('chunk')} (attempt {e.get('attempt')})"
    if kind == "chunk_finished":
        status = e.get("status", "")
        colour = C["green"] if status == "completed" else C["red"]
        extra = f" - {_clip(e.get('kill_reason', ''), 60)}" if e.get("kill_reason") else ""
        return (f"{stamp} {colour}{status.upper()}{C['off']} {e.get('chunk')} "
                f"in {e.get('turns')} turns{extra}{cost()}")
    if kind == "turn_budget_warning":
        return (f"{stamp} {C['yellow']}BUDGET{C['off']} {e.get('chunk', 'agent')} has "
                f"{e.get('turns_left')} turns left")
    if kind == "chunk_skipped":
        return f"{stamp} {C['yellow']}SKIP{C['off']} {e.get('chunk')} - {_clip(e.get('reason', ''), 60)}"
    if kind == "merge_conflict":
        return f"{stamp} {C['red']}CONFLICT{C['off']} {e.get('chunk')} - {_clip(e.get('reason', ''), 60)}"
    if kind == "run_end":
        return (f"{stamp} {C['bold']}DONE{C['off']} {e.get('succeeded')}/{e.get('total')} chunks, "
                f"merge={e.get('merge_commit') or 'none'}{cost()}")

    # explore
    if kind == "explore_split_done":
        return f"{stamp} {C['bold']}SPLIT{C['off']} {e.get('sub_question_count')} sub-question(s){cost()}"
    if kind == "explore_split_fallback":
        return f"{stamp} {C['yellow']}SPLIT FELL BACK{C['off']} to the original question"
    if kind == "explore_agent_start":
        return f"{stamp} {C['blue']}ASK{C['off']} {_clip(e.get('sub_question', ''), 80)}"
    if kind == "explore_agent_end":
        ok = e.get("stop_reason") == "finished" and e.get("answer_preview")
        colour = C["green"] if ok else C["red"]
        return (f"{stamp} {colour}{'ANSWERED' if ok else 'NO ANSWER'}{C['off']} "
                f"({e.get('stop_reason')}, {e.get('turns')} turns){cost()}")
    if kind == "explore_end":
        return f"{stamp} {C['bold']}DONE{C['off']} {e.get('answer_chars')} char answer{cost()}"

    if kind == "tool_call":
        preview = e.get("result_preview", "")
        if preview.startswith("error:"):
            return f"{stamp} {C['red']}TOOL FAIL{C['off']} {e.get('tool')}: {_clip(preview, 70)}"
        return None  # successful tool calls are the bulk of the log and rarely news
    if "error" in kind or "failed" in kind:
        return f"{stamp} {C['red']}{kind.upper()}{C['off']} {_clip(json.dumps(e), 90)}"
    if kind.endswith("_retry") or kind.endswith("_empty_turn"):
        return f"{stamp} {C['yellow']}{kind}{C['off']} {_clip(json.dumps(e), 80)}"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo", type=Path, help="Repository the run targets.")
    ap.add_argument("run_id", nargs="?", help="Run to follow. Default: the newest.")
    ap.add_argument("--scratch-dir", type=Path, default=None,
                    help="Defaults to <repo>/.pipeline-runs.")
    ap.add_argument("--once", action="store_true", help="Print and exit instead of following.")
    args = ap.parse_args()

    scratch = args.scratch_dir or (args.repo / ".pipeline-runs")
    if not scratch.exists():
        print(f"no runs yet under {scratch}", file=sys.stderr)
        return 1
    run_dir = (scratch / args.run_id) if args.run_id else _newest_run(scratch)
    if run_dir is None or not run_dir.exists():
        print(f"no run found under {scratch}", file=sys.stderr)
        return 1

    path = run_dir / "events.jsonl"
    print(f"{C['dim']}watching {path}{C['off']}", file=sys.stderr)
    totals = {"prompt": 0, "completion": 0, "calls": 0, "unreported": 0}
    offset = 0
    finished = False

    while True:
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                fh.seek(offset)
                for line in fh:
                    if not line.endswith("\n"):
                        break  # a torn final line; re-read it once it is complete
                    offset += len(line.encode("utf-8"))
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get("kind") in ("run_end", "explore_end"):
                        finished = True
                    rendered = _render(event, totals)
                    if rendered:
                        print(rendered, flush=True)
        if args.once or finished:
            return 0
        time.sleep(1.0)


if __name__ == "__main__":
    sys.exit(main())
