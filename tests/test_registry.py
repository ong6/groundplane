import pytest

from groundplane import FactRegistry, Ranking, UnregisteredFact


def test_record_captures_provenance():
    reg = FactRegistry()
    fact = reg.record("total", 42, tool="warehouse.query", args={"table": "orders"})
    assert fact.value == 42
    assert fact.provenance.tool == "warehouse.query"
    assert "table='orders'" in str(fact.provenance)
    assert "total" in reg and len(reg) == 1


def test_facts_are_write_once():
    reg = FactRegistry()
    reg.record("total", 42, tool="a")
    with pytest.raises(ValueError, match="write-once"):
        reg.record("total", 43, tool="b")


def test_unregistered_fact_lists_known_names():
    reg = FactRegistry()
    reg.record("total", 1, tool="a")
    with pytest.raises(UnregisteredFact) as exc:
        reg.value("missing")
    assert "total" in str(exc.value)


def test_ranking_is_computed_not_supplied(registry):
    ranking: Ranking = registry.ranking("campaign_ctr")
    assert ranking.winner == "harbour"
    assert ranking.names[:2] == ("harbour", "north")
    assert ranking.rank_of("north") == 2


def test_lower_is_better_ranking():
    reg = FactRegistry()
    reg.record_ranking(
        "latency_ms",
        {"eu": 120.0, "us": 88.0, "apac": 210.0},
        key="p99_latency_ms",
        tool="metrics.p99",
        higher_is_better=False,
    )
    assert reg.ranking("latency_ms").winner == "us"


def test_tool_decorator_registers_return_value():
    reg = FactRegistry()

    @reg.tool("warehouse.query", fact="row_count")
    def query(*, table: str) -> int:
        return 7

    assert query(table="orders") == 7
    assert reg.value("row_count") == 7
    assert reg.get("row_count").provenance.tool == "warehouse.query"


def test_ranking_accessor_rejects_plain_fact():
    reg = FactRegistry()
    reg.record("total", 42, tool="a")
    with pytest.raises(TypeError, match="not a Ranking"):
        reg.ranking("total")
