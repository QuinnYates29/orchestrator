"""Let plain `async def test_*` functions run without pytest-asyncio.

The repo has no async plugin, and its convention is a `_run(asyncio.run)`
helper defined inside each test module. `tests/test_pipeline_tools.py` was
written against the more common `async def test_*` style instead, so all 22
of its tests were collected and immediately reported as failures - they had
never executed once, which is how a broken `edit_file` no-op check shipped
unnoticed for the whole run.

Rather than mechanically rewriting those call sites, this hook makes the
style work: pytest hands us the test function, and if it is a coroutine
function we drive it ourselves. Modules using the `_run` helper are
unaffected, since those test functions are ordinary sync callables.
"""
import asyncio
import inspect

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    fn = pyfuncitem.obj
    if not inspect.iscoroutinefunction(fn):
        return None
    kwargs = {name: pyfuncitem.funcargs[name]
              for name in pyfuncitem._fixtureinfo.argnames}
    asyncio.run(fn(**kwargs))
    return True
