"""Runs in CI's adapter job with the actual optional framework packages installed."""

import asyncio
from typing import TypedDict

import pytest

from groundplane import FactRegistry, superlative
from groundplane.adapters.langgraph import guarded_node
from groundplane.adapters.mcp import payload_of, record_result


def test_actual_mcp_result_prefers_structured_content_and_refuses_errors():
    types = pytest.importorskip("mcp.types")
    result = types.CallToolResult(
        content=[types.TextContent(type="text", text='{"n": 999}')],
        structuredContent={"n": 1},
    )
    assert payload_of(result) == {"n": 1}
    reg = FactRegistry()
    record_result(reg, "n", result, tool="query", args={"window": "7d"})
    assert reg.value("n") == {"n": 1}
    error = types.CallToolResult(
        content=[types.TextContent(type="text", text="failed")], isError=True
    )
    with pytest.raises(ValueError, match="error result"):
        record_result(reg, "error", error, tool="query")
    assert "error" not in reg


@pytest.mark.parametrize("async_node", [False, True])
@pytest.mark.parametrize("winner", ["harbour", "north"])
def test_compiled_langgraph_routes_an_unsupported_claim_to_reask(async_node, winner):
    graph_module = pytest.importorskip("langgraph.graph")

    class State(TypedDict, total=False):
        winner: str
        reask: str
        route: str

    reg = FactRegistry()
    reg.record_ranking("ctr", {"harbour": 2, "north": 1}, key="ctr", tool="query")

    def summarize(state):
        return {"winner": winner}

    async def summarize_async(state):
        await asyncio.sleep(0)
        return summarize(state)

    node = guarded_node(
        summarize_async if async_node else summarize,
        registry=reg,
        facts=["ctr"],
        checks=[superlative(fact="ctr")],
        on_violation=lambda exc, state: {"reask": str(exc)},
    )
    graph = graph_module.StateGraph(State)
    graph.add_node("summarize", node)
    graph.add_node("retry", lambda state: {"route": "retry"})
    graph.add_node("publish", lambda state: {"route": "publish"})
    graph.set_entry_point("summarize")
    graph.add_conditional_edges(
        "summarize", lambda state: "retry" if "reask" in state else "publish"
    )
    graph.add_edge("retry", graph_module.END)
    graph.add_edge("publish", graph_module.END)
    compiled = graph.compile()
    result = asyncio.run(compiled.ainvoke({})) if async_node else compiled.invoke({})
    if winner == "north":
        assert result["route"] == "retry"
        assert "winner" not in result
        assert "argmax" in result["reask"]
    else:
        assert result == {"winner": "harbour", "route": "publish"}
