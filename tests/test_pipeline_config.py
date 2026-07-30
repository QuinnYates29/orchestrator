from __future__ import annotations

import textwrap

import pytest

from pipeline import config as pipeline_config
from pipeline.config import PipelineCfg, from_raw


def _write(tmp_path, body: str):
    target = tmp_path / "config.yaml"
    target.write_text(textwrap.dedent(body))
    return target


def test_defaults_when_block_absent(tmp_path):
    path = _write(tmp_path, """
        listen:
          host: 0.0.0.0
          port: 8080
    """)
    cfg = pipeline_config.load(path)
    assert cfg.roles.planner == "ds4-full"
    assert cfg.roles.worker == "ornith"
    assert cfg.limits.max_concurrent_workers == 4
    assert cfg.verify.command is None


def test_missing_file_yields_defaults(tmp_path):
    assert pipeline_config.load(tmp_path / "nope.yaml") == PipelineCfg()


def test_partial_block_overrides_only_what_it_names(tmp_path):
    path = _write(tmp_path, """
        pipeline:
          roles:
            worker: gemma
          limits:
            tool_output_chars: 1234
    """)
    cfg = pipeline_config.load(path)
    assert cfg.roles.worker == "gemma"
    assert cfg.roles.planner == "ds4-full"          # untouched default
    assert cfg.limits.tool_output_chars == 1234
    assert cfg.limits.max_agent_turns == 60         # untouched default


def test_unknown_role_key_is_an_error_not_silently_ignored():
    # A typo'd role would otherwise fall back to the default and be invisible
    # for the hours a run takes.
    with pytest.raises(ValueError, match="suprevisor"):
        from_raw({"roles": {"suprevisor": "ds4-light"}})


def test_unknown_top_level_key_is_an_error():
    with pytest.raises(ValueError, match="limitz"):
        from_raw({"limitz": {"max_agent_turns": 5}})


def test_verify_configured_is_false_for_blank_command():
    assert not from_raw({"verify": {"command": None}}).verify.configured
    assert not from_raw({"verify": {"command": "   "}}).verify.configured
    assert from_raw({"verify": {"command": "pytest -q"}}).verify.configured


def test_model_for_rejects_unknown_role():
    cfg = PipelineCfg()
    assert cfg.model_for("merger") == "ds4-full"
    with pytest.raises(KeyError):
        cfg.model_for("nonsense")


def test_repo_config_yaml_parses():
    """The real config.yaml shipped next to the package must load."""
    cfg = pipeline_config.load()
    assert cfg.roles.planner
    assert cfg.limits.max_concurrent_workers >= 1


# --- The whole `pipeline:` block must reach RunConfig, not just roles ---

from pathlib import Path as _Path

from pipeline.cli import build_parser, _build_run_config
from pipeline import config as _pc


def _run_config_from(config_text: str, tmp_path, extra_argv=()):
    cfg_file = tmp_path / "pipeline.yaml"
    cfg_file.write_text(config_text)
    task = tmp_path / "task.md"
    task.write_text("do the thing")
    argv = ["run", "--repo", str(tmp_path), "--task-file", str(task),
            "--config", str(cfg_file), *extra_argv]
    args = build_parser().parse_args(argv)
    return _build_run_config(args, _pc.load(cfg_file))


def test_verify_command_reaches_the_run(tmp_path):
    """`roles` was plumbed through and `verify` was not, so a configured verify
    command silently became None - which reports as *skipped*, i.e. as a
    deliberate choice rather than a dropped setting. The verify/repair loop was
    dead for every run."""
    rc = _run_config_from(
        "pipeline:\n  verify:\n    command: python3 -m pytest tests/ -q\n", tmp_path)
    assert rc.verify.command == "python3 -m pytest tests/ -q"
    assert rc.verify.configured is True


def test_verify_repair_attempts_reach_the_run(tmp_path):
    rc = _run_config_from(
        "pipeline:\n  verify:\n    command: make check\n    max_repair_attempts: 5\n", tmp_path)
    assert rc.verify.max_repair_attempts == 5


def test_limits_reach_the_run(tmp_path):
    rc = _run_config_from(
        "pipeline:\n  limits:\n    max_concurrent_workers: 2\n    tool_output_chars: 1234\n",
        tmp_path)
    assert rc.limits.max_concurrent_workers == 2
    assert rc.limits.tool_output_chars == 1234


def test_cli_flag_still_beats_the_configured_turn_ceiling(tmp_path):
    rc = _run_config_from(
        "pipeline:\n  limits:\n    max_agent_turns: 60\n", tmp_path,
        extra_argv=["--max-agent-turns", "40"])
    assert rc.max_agent_turns == 40


def test_configured_turn_ceiling_applies_without_the_flag(tmp_path):
    rc = _run_config_from("pipeline:\n  limits:\n    max_agent_turns: 25\n", tmp_path)
    assert rc.max_agent_turns == 25


def test_no_verify_configured_is_still_none(tmp_path):
    """Absence must stay absent - this fix must not invent a default command."""
    rc = _run_config_from("pipeline:\n  roles:\n    planner: ds4\n", tmp_path)
    assert rc.verify.command is None
    assert rc.verify.configured is False
