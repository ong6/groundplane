import pytest

from groundplane import FactRegistry, Ranking, Table, UnregisteredFact


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


# A duplicate key means a named row is ambiguous — rows.py resolves a row by this key,
# so the table must refuse to exist rather than let a lookup pick the first match.
def test_record_table_rejects_duplicate_key():
    reg = FactRegistry()
    with pytest.raises(ValueError, match="does not identify a row"):
        reg.record_table(
            "campaign_rows",
            [{"campaign": "north", "ctr": 0.04}, {"campaign": "north", "ctr": 0.05}],
            key="campaign",
            tool="warehouse.query",
        )


# A row that dropped the key column entirely: the tool returned a partial record.
def test_record_table_rejects_row_missing_key():
    reg = FactRegistry()
    with pytest.raises(ValueError, match="no key column"):
        reg.record_table(
            "campaign_rows",
            [{"campaign": "north", "ctr": 0.04}, {"ctr": 0.05}],
            key="campaign",
            tool="warehouse.query",
        )


# A key naming a column nobody returned — including a columns= list that omits it.
def test_record_table_rejects_key_outside_columns():
    reg = FactRegistry()
    with pytest.raises(ValueError, match="not among the table columns"):
        reg.record_table(
            "campaign_rows",
            [{"campaign": "north", "ctr": 0.04}],
            key="campaign_id",
            tool="warehouse.query",
        )
    with pytest.raises(ValueError, match="not among the table columns"):
        reg.record_table(
            "campaign_rows",
            [{"campaign": "north", "ctr": 0.04}],
            key="campaign",
            tool="warehouse.query",
            columns=["ctr"],
        )


# Integer row ids must survive ranking: a model answering 101 is compared against
# the int 101, never the string "101".
def test_to_ranking_preserves_key_type():
    reg = FactRegistry()
    reg.record_table(
        "campaign_rows",
        [{"id": 101, "ctr": 0.0455}, {"id": 102, "ctr": 0.0412}, {"id": 103, "ctr": 0.0301}],
        key="id",
        tool="warehouse.query",
    )
    ranking = reg.table("campaign_rows").to_ranking("ctr")
    assert ranking.winner == 101
    assert isinstance(ranking.winner, int)
    assert ranking.names == (101, 102, 103)
    assert all(isinstance(name, int) for name in ranking.names)
    assert ranking.score_of(101) == pytest.approx(0.0455)
    assert ranking.score_of("101") is None


# The same for a directly recorded ranking: int names round-trip through every accessor.
def test_record_ranking_round_trips_int_names():
    reg = FactRegistry()
    reg.record_ranking(
        "campaign_ctr",
        {101: 0.0455, 102: 0.0412, 103: 0.0301},
        key="ctr",
        tool="warehouse.query",
    )
    ranking = reg.ranking("campaign_ctr")
    assert ranking.winner == 101
    assert ranking.names == (101, 102, 103)
    assert ranking.rank_of(102) == 2
    assert ranking.score_of(103) == pytest.approx(0.0301)
    assert ranking.rank_of("102") is None


# One typed accessor, one message: the wrong type must name what it got and how to fix it.
def test_as_type_is_the_single_typed_accessor(registry):
    with pytest.raises(TypeError) as exc:
        registry.table("campaign_ctr")
    msg = str(exc.value)
    assert "Ranking" in msg and "not a Table" in msg and "record_table()" in msg

    registry.record("total", 42, tool="a")
    with pytest.raises(TypeError, match="not a Ranking or Table"):
        registry.as_type("total", Ranking, Table)
