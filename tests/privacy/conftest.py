import pytest

from mmore.privacy.agents.base import clear_llm_cache
from mmore.privacy.agents.registry import tool_registry


@pytest.fixture(autouse=True)
def _isolate_llm_cache():
    clear_llm_cache()
    yield
    clear_llm_cache()


@pytest.fixture
def isolated_tool_registry():
    snapshot = dict(tool_registry)
    tool_registry.clear()
    try:
        yield
    finally:
        tool_registry.clear()
        tool_registry.update(snapshot)
