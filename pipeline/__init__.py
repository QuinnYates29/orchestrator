"""Multi-agent implementation pipeline built on top of the orchestrator's
OpenAI-compatible API: ds4-full plans and later reviews/merges; N Ornith
agents implement plan chunks in parallel, each in an isolated local git
clone; ds4-light supervises the Ornith agents concurrently, watching for
stuck-thinking loops and killing/retrying them.

This package is a pure HTTP client of the orchestrator - it does not import
from the `orchestrator` package and assumes the orchestrator server is
already running and reachable.
"""
