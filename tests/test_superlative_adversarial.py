"""Adversarial fixtures: plausible model answers that are provably wrong.

Each case names the real-world failure it stands for.
"""

import pytest

from groundplane import (
    FactRegistry,
    UnsupportedClaim,
    boundary,
    check_superlative,
    field_matches_fact,
    superlative,
)


def _run(registry, output, **kw):
    with boundary(
        registry, facts=["campaign_ctr"], checks=[superlative(fact="campaign_ctr", **kw)]
    ) as b:
        b.submit(output)


# 1. The motivating failure: the model names the famous campaign, not the argmax.
def test_wrong_winner_raises(registry):
    with pytest.raises(UnsupportedClaim) as exc:
        _run(registry, {"winner": "north"})
    msg = str(exc.value)
    assert "'north'" in msg and "'harbour'" in msg
    assert "warehouse.query" in msg  # provenance is in the error
    assert "#2" in msg
    assert exc.value.field == "winner"
    assert exc.value.supported == "harbour"


# 2. Right winner, invented metric. The number reads authoritative and is not the data.
def test_right_winner_wrong_score_raises(registry):
    with pytest.raises(UnsupportedClaim, match="does not match"):
        _run(registry, {"winner": "harbour", "ctr": 0.061}, score_field="ctr")


# 3. Hallucinated entity: a campaign that was never in the result set.
def test_unranked_entity_raises(registry):
    with pytest.raises(UnsupportedClaim, match="never ranked"):
        _run(registry, {"winner": "meridian"})


# 4. A genuine tie flattened into a single confident winner.
def test_tie_reported_as_sole_winner_raises():
    reg = FactRegistry()
    reg.record_ranking(
        "campaign_ctr",
        {"north": 0.04, "harbour": 0.04, "delta": 0.01},
        key="ctr",
        tool="warehouse.query",
    )
    with pytest.raises(UnsupportedClaim, match="tie"):
        _run(reg, {"winner": "north"})
    # ...and is allowed only when the caller explicitly permits picking one.
    _run(reg, {"winner": "north"}, allow_tie_pick=True)


# 5. Direction flip: "best" latency means lowest, the model picked the largest number.
def test_lower_is_better_direction_flip_raises():
    reg = FactRegistry()
    reg.record_ranking(
        "campaign_ctr",
        {"eu": 120.0, "us": 88.0, "apac": 210.0},
        key="p99_latency_ms",
        tool="metrics.p99",
        higher_is_better=False,
    )
    with pytest.raises(UnsupportedClaim) as exc:
        _run(reg, {"winner": "apac"})
    assert "'us'" in str(exc.value)


# 6. The field is simply absent: an unverifiable claim is a failed claim.
def test_missing_declared_field_raises(registry):
    with pytest.raises(UnsupportedClaim, match="no such field"):
        _run(registry, {"headline": "Harbour led the week."})


# 7. Score smuggled in as a string that looks numeric.
def test_non_numeric_score_raises(registry):
    with pytest.raises(UnsupportedClaim, match="not a number"):
        _run(registry, {"winner": "harbour", "ctr": "0.0455"}, score_field="ctr")


# 8. Correct winner, wrong rank asserted alongside it.
def test_inconsistent_rank_field_raises(registry):
    with pytest.raises(UnsupportedClaim, match="rank must be 1"):
        _run(registry, {"winner": "harbour", "rank": 2}, rank_field="rank")


def test_score_tolerance_allows_rounding(registry):
    _run(registry, {"winner": "harbour", "ctr": 0.046}, score_field="ctr", tolerance=0.001)


def test_field_matches_fact_guards_plain_values():
    reg = FactRegistry()
    reg.record("row_count", 1042, tool="warehouse.query", args={"table": "orders"})
    check = field_matches_fact(field="rows", fact="row_count")
    check(reg, {"rows": 1042})
    with pytest.raises(UnsupportedClaim) as exc:
        check(reg, {"rows": 1000})
    assert exc.value.supported == 1042


# 9. The superlative guard reads a recorded Table the same way ranking_prefix does:
# one recorded fact, both checks, and the wrong winner still raises.
def test_superlative_over_a_table_fact():
    reg = FactRegistry()
    reg.record_table(
        "campaign_rows",
        [
            {"campaign": "north", "ctr": 0.0412},
            {"campaign": "harbour", "ctr": 0.0455},
            {"campaign": "delta", "ctr": 0.0301},
        ],
        key="campaign",
        tool="warehouse.query",
    )
    check_superlative(reg, {"winner": "harbour"}, fact="campaign_rows", column="ctr")
    with pytest.raises(UnsupportedClaim, match="computed argmax"):
        check_superlative(reg, {"winner": "north"}, fact="campaign_rows", column="ctr")
    with pytest.raises(ValueError, match="pass column="):
        check_superlative(reg, {"winner": "harbour"}, fact="campaign_rows")
