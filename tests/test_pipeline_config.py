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
