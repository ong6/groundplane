"""Adversarial fixtures for ranking_prefix: plausible top-k lists that are wrong.

Each case names the real-world failure it stands for.
"""

import pytest

from groundplane import FactRegistry, UnsupportedClaim, boundary, superlative
from groundplane.ordering import check_ranking_prefix, ranking_prefix

# Same shape as the superlative fixtures: "north" is the famous campaign, "harbour"
# is the argmax, and "delta" is the plausible-sounding interloper that is really #4.
CTR = {"north": 0.0412, "harbour": 0.0455, "delta": 0.0301, "vega": 0.0330}


@pytest.fixture
def registry() -> FactRegistry:
    reg = FactRegistry()
    reg.record_ranking(
        "campaign_ctr",
        CTR,
        key="ctr",
        tool="warehouse.query",
        args={"table": "campaign_daily", "window": "7d"},
    )
    return reg


@pytest.fixture
def table_registry() -> FactRegistry:
    reg = FactRegistry()
    reg.record_table(
        "campaign_rows",
        [{"campaign": name, "ctr": ctr} for name, ctr in CTR.items()],
        key="campaign",
        tool="warehouse.query",
        args={"table": "campaign_daily", "window": "7d"},
    )
    return reg


def _run(reg, output, *, fact="campaign_ctr", **kw):
    with boundary(reg, facts=[fact], checks=[ranking_prefix(fact=fact, field="top", **kw)]) as b:
        b.submit(output)


def _tied_registry(scores):
    reg = FactRegistry()
    reg.record_ranking("campaign_ctr", scores, key="ctr", tool="warehouse.query")
    return reg


# 1. The set is right, the order is invented. Order is the whole claim under ordered=True.
def test_right_set_wrong_order_raises(registry):
    with pytest.raises(UnsupportedClaim) as exc:
        _run(registry, {"top": ["north", "harbour", "delta"]}, k=3)
    assert "'north'" in str(exc.value)
    assert "warehouse.query" in str(exc.value)
    # ...and the same list is fine when the caller only claimed a set.
    _run(registry, {"top": ["north", "harbour", "vega"]}, k=3, ordered=False)


# 2. A plausible name slipped into the last slot; the real #3 is dropped silently.
def test_interloper_at_position_three_raises(registry):
    with pytest.raises(UnsupportedClaim) as exc:
        _run(registry, {"top": ["harbour", "north", "delta"]}, k=3)
    msg = str(exc.value)
    assert "'delta'" in msg and "#4" in msg
    assert "0.0301" in msg


# 3. Four names under a "top 3" heading: the length is a claim of its own.
def test_length_mismatch_under_top_three_raises(registry):
    with pytest.raises(UnsupportedClaim, match="top-3"):
        _run(registry, {"top": ["harbour", "north", "vega", "delta"]}, k=3)


# 4. A genuine tie reordered is not an error. The check's main false-positive risk.
def test_legitimate_tie_reordered_passes():
    reg = _tied_registry({"harbour": 0.05, "north": 0.04, "vega": 0.04, "delta": 0.01})
    _run(reg, {"top": ["harbour", "north", "vega"]}, k=3)
    _run(reg, {"top": ["harbour", "vega", "north"]}, k=3)


# 5. "Top 3" when #3 and #4 share a score: the cut is not a fact.
def test_cut_inside_tie_block_raises():
    reg = _tied_registry({"harbour": 0.05, "north": 0.04, "vega": 0.03, "delta": 0.03})
    with pytest.raises(UnsupportedClaim, match="tie"):
        _run(reg, {"top": ["harbour", "north", "vega"]}, k=3)
    _run(reg, {"top": ["harbour", "north", "vega"]}, k=3, allow_tie_pick=True)
    _run(reg, {"top": ["harbour", "north", "delta"]}, k=3, allow_tie_pick=True)


# 6. Padding the list to hit the requested length by repeating the winner.
def test_duplicate_padding_raises(registry):
    with pytest.raises(UnsupportedClaim, match="twice"):
        _run(registry, {"top": ["harbour", "harbour", "north"]}, k=3)


# 7. A bare string must not be iterated character by character into a "ranking".
def test_bare_string_instead_of_list_raises(registry):
    with pytest.raises(UnsupportedClaim, match="not a list"):
        _run(registry, {"top": "harbour"}, k=1)


# 8. The same verdicts driven off a recorded Table rather than a Ranking.
def test_table_backed_ranking_agrees(table_registry):
    _run(table_registry, {"top": ["harbour", "north"]}, fact="campaign_rows", k=2, column="ctr")
    with pytest.raises(UnsupportedClaim) as exc:
        _run(
            table_registry,
            {"top": ["harbour", "north", "delta"]},
            fact="campaign_rows",
            k=3,
            column="ctr",
        )
    assert "'delta'" in str(exc.value) and "#4" in str(exc.value)


# 9. k=1 is the strict generalisation of the superlative guard.
def test_k1_agrees_with_superlative(registry):
    output = {"top": ["north"], "winner": "north"}
    with pytest.raises(UnsupportedClaim):
        check_ranking_prefix(registry, output, fact="campaign_ctr", field="top", k=1)
    with pytest.raises(UnsupportedClaim):
        superlative(fact="campaign_ctr")(registry, output)


# Misconfiguration is a developer bug, and must not read as a model failure.
def test_column_on_a_ranking_fact_is_a_config_error(registry):
    with pytest.raises(ValueError, match="already a Ranking"):
        check_ranking_prefix(
            registry, {"top": ["harbour"]}, fact="campaign_ctr", field="top", column="ctr"
        )


def test_missing_declared_field_raises(registry):
    with pytest.raises(UnsupportedClaim, match="no such field"):
        _run(registry, {"headline": "Harbour led the week."}, k=3)


def test_happy_path_open_length(registry):
    _run(registry, {"top": ["harbour", "north"]})


@pytest.fixture
def int_key_registry() -> FactRegistry:
    reg = FactRegistry()
    reg.record_table(
        "campaign_rows",
        [{"id": 101, "ctr": 0.0455}, {"id": 102, "ctr": 0.0412}, {"id": 103, "ctr": 0.0301}],
        key="id",
        tool="warehouse.query",
        args={"table": "campaign_daily", "window": "7d"},
    )
    return reg


# 10. Integer row keys stay integers through to the comparison: the model returning 101
# must match, and the stringified "101" must not quietly pass as the same entity.
def test_int_row_keys_are_not_stringified(int_key_registry):
    _run(int_key_registry, {"top": [101, 102]}, fact="campaign_rows", k=2, column="ctr")
    with pytest.raises(UnsupportedClaim, match="never ranked"):
        _run(int_key_registry, {"top": ["101", "102"]}, fact="campaign_rows", k=2, column="ctr")


# 11. Same gap at k=1: the superlative guard compares winners by value, not by str().
def test_superlative_on_int_keys():
    reg = FactRegistry()
    reg.record_ranking(
        "campaign_ctr",
        {101: 0.0455, 102: 0.0412, 103: 0.0301},
        key="ctr",
        tool="warehouse.query",
    )
    superlative(fact="campaign_ctr")(reg, {"winner": 101})
    with pytest.raises(UnsupportedClaim, match="never ranked"):
        superlative(fact="campaign_ctr")(reg, {"winner": "101"})
