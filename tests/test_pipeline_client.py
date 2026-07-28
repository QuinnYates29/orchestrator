from __future__ import annotations

from pipeline.client import OrchestratorClient, _ToolCallAccumulator


def test_build_body_includes_tools_and_tool_choice():
    client = OrchestratorClient.__new__(OrchestratorClient)  # skip __init__, no real httpx client needed
    body = client._build_body(
        "ornith", [{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "x"}}],
        tool_choice="auto", max_tokens=100, temperature=0.5, stream=True,
    )
    assert body["model"] == "ornith"
    assert body["stream"] is True
    assert body["tools"][0]["function"]["name"] == "x"
    assert body["tool_choice"] == "auto"
    assert body["max_tokens"] == 100
    assert body["temperature"] == 0.5


def test_build_body_omits_unset_optional_fields():
    client = OrchestratorClient.__new__(OrchestratorClient)
    body = client._build_body(
        "ornith", [{"role": "user", "content": "hi"}],
        tools=None, tool_choice=None, max_tokens=None, temperature=None, stream=False,
    )
    assert "tools" not in body
    assert "tool_choice" not in body
    assert "max_tokens" not in body
    assert "temperature" not in body


def test_build_body_forwards_extra_kwargs():
    client = OrchestratorClient.__new__(OrchestratorClient)
    body = client._build_body(
        "ornith", [], tools=None, tool_choice=None, max_tokens=None, temperature=None,
        stream=False, reasoning_effort="high",
    )
    assert body["reasoning_effort"] == "high"


def test_tool_call_accumulator_finalizes_valid_json():
    acc = _ToolCallAccumulator(id="call_1", name="read_file", arguments_buf='{"path": "a.py"}')
    result = acc.finalize()
    assert result == {"id": "call_1", "name": "read_file", "arguments": {"path": "a.py"}}


def test_tool_call_accumulator_handles_streamed_fragments():
    # Real streaming delivers the arguments string in fragments across
    # multiple deltas - simulate that by building it up incrementally.
    acc = _ToolCallAccumulator()
    acc.id = "call_2"
    acc.name = "write_file"
    for fragment in ['{"path"', ': "a.py",', ' "content": "x"}']:
        acc.arguments_buf += fragment
    result = acc.finalize()
    assert result["arguments"] == {"path": "a.py", "content": "x"}


def test_tool_call_accumulator_falls_back_on_malformed_json():
    acc = _ToolCallAccumulator(id="call_3", name="read_file", arguments_buf="{not valid json")
    result = acc.finalize()
    assert result["arguments"] == {"_raw": "{not valid json"}


def test_tool_call_accumulator_empty_arguments():
    acc = _ToolCallAccumulator(id="call_4", name="list_dir", arguments_buf="")
    result = acc.finalize()
    assert result["arguments"] == {}
