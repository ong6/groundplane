"""Adapter tests. Neither adapter imports its framework, so neither test needs it.

The fakes here are the shapes the frameworks actually pass: a LangGraph node is a
callable taking state and returning a partial update; an MCP result is an object
carrying structured content or text blocks.
"""

import asyncio

import pytest

from groundplane import FactRegistry, UnsupportedClaim, superlative
from groundplane.adapters.langgraph import guarded_node, record_state
from groundplane.adapters.mcp import call_and_record, payload_of, record_result

CHECKS = [superlative(fact="campaign_ctr")]


def _summarise(winner):
    def node(state):
        return {"summary": {"winner": winner}, "steps": state.get("steps", 0) + 1}

    return node


# 1. A node whose update names the argmax passes through untouched.
def test_guarded_node_passes_the_correct_winner(registry):
    node = guarded_node(
        _summarise("harbour"),
        registry=registry,
        facts=["campaign_ctr"],
        checks=CHECKS,
        output_key="summary",
    )
    assert node({"steps": 0}) == {"summary": {"winner": "harbour"}, "steps": 1}


# 2. The wrong winner kills the run by default. An unchecked answer must not flow on.
def test_guarded_node_raises_by_default(registry):
    node = guarded_node(
        _summarise("north"),
        registry=registry,
        facts=["campaign_ctr"],
        checks=CHECKS,
        output_key="summary",
    )
    with pytest.raises(UnsupportedClaim, match="computed argmax"):
        node({"steps": 0})


# 3. The reask edge: on_violation converts the failure into a state update the
# graph can route on, and the offending update never reaches the state.
def test_guarded_node_routes_to_reask(registry):
    node = guarded_node(
        _summarise("north"),
        registry=registry,
        facts=["campaign_ctr"],
        checks=CHECKS,
        output_key="summary",
        on_violation=lambda exc, state: {"reask": str(exc), "steps": state["steps"]},
    )
    update = node({"steps": 3})
    assert update["steps"] == 3
    assert "summary" not in update
    assert "computed argmax" in update["reask"]


# 4. A developer bug is not something to reask the model about, so a misconfigured
# check propagates even when a reask handler is installed.
def test_guarded_node_does_not_swallow_config_errors(registry):
    node = guarded_node(
        _summarise("harbour"),
        registry=registry,
        facts=["campaign_ctr"],
        checks=[superlative(fact="campaign_ctr", column="ctr")],
        output_key="summary",
        on_violation=lambda exc, state: {"reask": str(exc)},
    )
    with pytest.raises(ValueError, match="already a Ranking"):
        node({"steps": 0})


# 5. Without output_key the whole update is the checked payload.
def test_guarded_node_checks_the_whole_update(registry):
    node = guarded_node(
        lambda state: {"winner": "harbour"},
        registry=registry,
        facts=["campaign_ctr"],
        checks=CHECKS,
    )
    assert node({}) == {"winner": "harbour"}


# 6. Values a node upstream already put in the state can be adopted as facts.
def test_record_state_adopts_upstream_values():
    reg = FactRegistry()
    record_state(reg, {"rows": 7}, keys={"rows": "row_count"}, tool="warehouse.query")
    assert reg.value("row_count") == 7
    assert reg.get("row_count").provenance.tool == "warehouse.query"
    with pytest.raises(KeyError, match="no key 'missing'"):
        record_state(reg, {"rows": 7}, keys={"missing": "x"}, tool="t")


# --- MCP ---------------------------------------------------------------------


class _Text:
    type = "text"

    def __init__(self, text):
        self.text = text


class _Result:
    def __init__(self, *, structured=None, texts=(), is_error=False):
        self.structuredContent = structured
        self.content = [_Text(t) for t in texts]
        self.isError = is_error


# 7. Structured content wins over the text rendering of the same value.
def test_payload_prefers_structured_content():
    result = _Result(structured={"rows": [{"campaign": "harbour"}]}, texts=['{"rows": []}'])
    assert payload_of(result) == {"rows": [{"campaign": "harbour"}]}


# 8. A server that only returns JSON text still yields a value, not a blob.
def test_payload_parses_json_text():
    assert payload_of(_Result(texts=['{"ctr": 0.0455}'])) == {"ctr": 0.0455}
    assert payload_of(_Result(texts=["harbour led the week"])) == "harbour led the week"
    assert payload_of(_Result(texts=['{"ctr": 1}']), parse_json=False) == '{"ctr": 1}'


# 9. An error result is not a fact, and neither is an empty one.
def test_error_and_empty_results_are_not_facts():
    with pytest.raises(ValueError, match="error result"):
        payload_of(_Result(texts=["boom"], is_error=True))
    with pytest.raises(ValueError, match="neither structured content nor text"):
        payload_of(_Result())


# 10. Recording a result keeps the call that produced it.
def test_record_result_keeps_provenance():
    reg = FactRegistry()
    reg_fact = record_result(
        reg,
        "campaign_rows",
        _Result(structured=[{"campaign": "harbour"}]),
        tool="warehouse.query",
        args={"window": "7d"},
    )
    assert reg_fact.value == [{"campaign": "harbour"}]
    assert str(reg.get("campaign_rows").provenance) == "warehouse.query(window='7d')"


# 11. The async path records the arguments actually sent. asyncio.run keeps this
# free of a pytest-asyncio dependency, which the dev extra does not need to carry.
def test_call_and_record_records_what_was_sent():
    import asyncio

    class _Session:
        def __init__(self):
            self.calls = []

        async def call_tool(self, name, arguments):
            self.calls.append((name, arguments))
            return _Result(structured={"ctr": 0.0455})

    reg = FactRegistry()
    session = _Session()
    asyncio.run(
        call_and_record(
            session, reg, fact="ctr", tool="warehouse.query", arguments={"window": "7d"}
        )
    )
    assert session.calls == [("warehouse.query", {"window": "7d"})]
    assert reg.value("ctr") == {"ctr": 0.0455}
    assert str(reg.get("ctr").provenance) == "warehouse.query(window='7d')"


def test_mcp_dictionary_payload_and_error_flags_follow_the_object_contract():
    result = {"structuredContent": {"n": 1}, "content": [{"type": "text", "text": '{"n": 2}'}]}
    assert payload_of(result) == {"n": 1}
    assert payload_of({"content": [{"type": "text", "text": '{"n": 2}'}]}) == {"n": 2}
    result["isError"] = True
    reg = FactRegistry()
    with pytest.raises(ValueError, match="error result"):
        record_result(reg, "bad", result, tool="query")
    assert len(reg) == 0


@pytest.mark.parametrize(
    "text", ['{"n": NaN}', '{"n": Infinity}', '{"n": -Infinity}', '{"n": 1e400}']
)
def test_mcp_nonfinite_json_cannot_become_recorded_evidence(text):
    reg = FactRegistry()
    with pytest.raises(ValueError, match="non-finite"):
        record_result(reg, "bad", _Result(texts=[text]), tool="query")
    assert len(reg) == 0


def test_mcp_malformed_text_block_is_a_data_error():
    with pytest.raises(ValueError, match="string"):
        payload_of({"content": [{"type": "text", "text": None}]})


def test_mcp_provenance_keeps_the_arguments_sent_before_the_await():
    arguments = {"filter": {"window": "7d"}}

    class Session:
        async def call_tool(self, name, sent):
            assert sent == {"filter": {"window": "7d"}}
            arguments["filter"]["window"] = "1d"
            sent["filter"]["window"] = "30d"
            await asyncio.sleep(0)
            return _Result(structured={"n": 1})

    reg = FactRegistry()
    asyncio.run(call_and_record(Session(), reg, fact="n", tool="query", arguments=arguments))
    assert reg.get("n").provenance.args == {"filter": {"window": "7d"}}


@pytest.mark.parametrize("winner", ["harbour", "north"])
@pytest.mark.parametrize("callable_object", [False, True])
def test_async_nodes_are_awaited_before_their_claim_crosses_the_boundary(
    registry, winner, callable_object
):
    async def summarize(state):
        await asyncio.sleep(0)
        return {"winner": winner}

    class Summarize:
        async def __call__(self, state):
            return await summarize(state)

    fn = Summarize() if callable_object else summarize
    node = guarded_node(fn, registry=registry, facts=["campaign_ctr"], checks=CHECKS)
    if winner == "harbour":
        assert asyncio.run(node({})) == {"winner": "harbour"}
    else:
        with pytest.raises(UnsupportedClaim):
            asyncio.run(node({}))


def test_async_nodes_keep_reask_and_configuration_failure_separate(registry):
    async def summarize(state):
        return {"summary": {"winner": "north"}}

    options = {
        "registry": registry,
        "facts": ["campaign_ctr"],
        "output_key": "summary",
        "on_violation": lambda exc, state: {"reask": str(exc)},
    }
    node = guarded_node(summarize, checks=CHECKS, **options)
    assert "reask" in asyncio.run(node({}))
    broken = guarded_node(
        summarize, checks=[superlative(fact="campaign_ctr", column="ctr")], **options
    )
    with pytest.raises(ValueError, match="already a Ranking"):
        asyncio.run(broken({}))
