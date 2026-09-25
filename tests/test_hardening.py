"""Adversarial failures at the evidence boundary, including valid JSON type confusion."""

import json
import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from groundplane import (
    FactBoundaryError,
    FactRegistry,
    UnregisteredFact,
    UnsupportedClaim,
    aggregate_reconciles,
    boundary,
    comparison,
    field_matches_fact,
    ranking_prefix,
    row_integrity,
    superlative,
)
from groundplane.ordering import check_ranking_prefix


def test_tool_input_and_every_public_read_cannot_rewrite_recorded_evidence():
    reg = FactRegistry()
    value = {"rows": [{"score": 1}]}
    args = {"filter": {"window": "7d"}}
    recorded = reg.record("f", value, tool="query", args=args)
    value["rows"][0]["score"] = 999
    args["filter"]["window"] = "1d"
    reads = [recorded, reg.get("f"), next(iter(reg))]
    for fact in reads:
        fact.value["rows"][0]["score"] = 999
        fact.provenance.args["filter"]["window"] = "1d"
    reg.value("f")["rows"][0]["score"] = 999
    assert reg.value("f") == {"rows": [{"score": 1}]}
    assert reg.get("f").provenance.args == {"filter": {"window": "7d"}}


def test_table_access_and_prompt_builder_cannot_rewrite_winner():
    reg = FactRegistry()
    rows = [{"id": "a", "score": 1, "meta": {"n": 1}}, {"id": "b", "score": 2}]
    reg.record_table("t", rows, key="id", tool="query")
    rows[0]["meta"]["n"] = 9
    reg.table("t").rows[0]["score"] = 999
    with boundary(reg, facts=["t"], checks=[superlative(fact="t", column="score")]) as b:
        b.facts()["t"].rows[0]["score"] = 999
        assert b.report({"winner": "a"}).ok is False
    assert reg.table("t").rows[0]["meta"] == {"n": 1}


@pytest.mark.parametrize("mode", ["first", "all", "report"])
def test_checks_cannot_approve_a_fact_from_outside_the_declared_scope(mode):
    reg = FactRegistry()
    reg.record("allowed", 1, tool="query")
    reg.record("stale", 999, tool="yesterday")
    with (
        pytest.raises(UnregisteredFact),
        boundary(reg, facts=["allowed"], checks=[field_matches_fact(field="n", fact="stale")]) as b,
    ):
        if mode == "report":
            b.report({"n": 999})
        else:
            b.submit({"n": 999}, mode=mode)


def test_custom_checks_only_see_declared_facts_and_cannot_register_new_evidence():
    reg = FactRegistry()
    reg.record("allowed", 1, tool="query")
    reg.record("secret", 2, tool="query")

    def check(scoped, output):
        assert scoped.names() == ["allowed"]
        assert "secret" not in scoped
        assert [f.name for f in scoped] == ["allowed"]
        assert len(scoped) == 1
        with pytest.raises(FactBoundaryError, match="read-only"):
            scoped.record("invented", output["n"], tool="model")

    with boundary(reg, facts=["allowed"], checks=[check]) as b:
        b.submit({"n": 1})
    assert "invented" not in reg


@pytest.mark.parametrize("bad", [True, 1.0, [1], {"id": 1}, (True,)])
@pytest.mark.parametrize("family", ["winner", "prefix", "row", "comparison"])
def test_boolean_float_or_container_cannot_impersonate_an_integer_entity(bad, family):
    reg = FactRegistry()
    reg.record_ranking("r", {1: 10, 2: 9}, key="n", tool="query")
    reg.record_table("t", [{"id": 1, "n": 10}, {"id": 2, "n": 9}], key="id", tool="query")
    checks = {
        "winner": (superlative(fact="r"), {"winner": bad}),
        "prefix": (ranking_prefix(fact="r", field="top"), {"top": [bad]}),
        "row": (row_integrity(fact="t", key_field="id", fields={"n": "n"}), {"id": bad, "n": 10}),
        "comparison": (
            comparison(fact="r", a_field="a", b_field="b", delta_field="d"),
            {"a": bad, "b": 2, "d": 1},
        ),
    }
    check, output = checks[family]
    with pytest.raises(UnsupportedClaim):
        check(reg, output)


@pytest.mark.parametrize("rank", [True, 1.0, "1"])
def test_first_place_requires_an_integer_rank(rank):
    reg = FactRegistry()
    reg.record_ranking("r", {"a": 10}, key="n", tool="query")
    with pytest.raises(UnsupportedClaim):
        superlative(fact="r", rank_field="rank")(reg, {"winner": "a", "rank": rank})


@pytest.mark.parametrize(
    "recorded,claimed", [(1, True), ({"n": [1]}, {"n": [True]}), ({1: "a"}, {True: "a"})]
)
def test_structural_fact_equality_does_not_turn_boolean_values_into_numbers(recorded, claimed):
    reg = FactRegistry()
    reg.record("f", recorded, tool="query")
    with pytest.raises(UnsupportedClaim):
        field_matches_fact(fact="f", field="value")(reg, {"value": claimed})


@given(st.integers(min_value=2**53, max_value=2**80))
def test_adjacent_large_integer_scores_remain_distinguishable(n):
    reg = FactRegistry()
    reg.record_table("t", [{"id": "a", "n": n}, {"id": "b", "n": n + 1}], key="id", tool="query")
    assert reg.table("t").to_ranking("n").winner == "b"
    assert reg.table("t").to_ranking("n").ties == ("b",)
    for fact in ("t", "r"):
        if fact == "r":
            reg.record_ranking("r", {"a": n, "b": n + 1}, key="n", tool="query")
        kwargs = {"column": "n"} if fact == "t" else {}
        check = superlative(fact=fact, score_field="score", **kwargs)
        check(reg, {"winner": "b", "score": n + 1})
        with pytest.raises(UnsupportedClaim):
            check(reg, {"winner": "b", "score": n})
    delta = comparison(fact="t", column="n", a_field="a", b_field="b", delta_field="d")
    delta(reg, {"a": "b", "b": "a", "d": 1})
    with pytest.raises(UnsupportedClaim):
        delta(reg, {"a": "b", "b": "a", "d": 0})


def test_integer_total_does_not_drop_a_unit_above_float_precision():
    reg = FactRegistry()
    reg.record_table("t", [{"id": "a", "n": 2**53}, {"id": "b", "n": 1}], key="id", tool="query")
    check = aggregate_reconciles(fact="t", field="n", column="n")
    check(reg, {"n": 2**53 + 1})
    with pytest.raises(UnsupportedClaim):
        check(reg, {"n": 2**53})


@pytest.mark.parametrize(
    "op,rows,kwargs",
    [
        ("median", [{"id": "a", "n": 1e308}, {"id": "b", "n": 1e308}], {}),
        ("mean", [{"id": "a", "n": 1e308, "w": 2}], {"weight_column": "w"}),
    ],
)
def test_overflow_cannot_support_an_infinite_aggregate(op, rows, kwargs):
    reg = FactRegistry()
    reg.record_table("t", rows, key="id", tool="query")
    check = aggregate_reconciles(fact="t", field="n", column="n", op=op, **kwargs)
    check(reg, {"n": 1e308})
    with pytest.raises(UnsupportedClaim):
        check(reg, {"n": math.inf})


@pytest.mark.parametrize("a,b", [(math.inf, 1), (1, math.inf)])
def test_nonfinite_cells_or_overflow_never_support_a_comparison(a, b):
    reg = FactRegistry()
    reg.record_table("t", [{"id": "a", "n": a}, {"id": "b", "n": b}], key="id", tool="query")
    with pytest.raises(ValueError, match="not finite"):
        comparison(fact="t", column="n", a_field="a", b_field="b", delta_field="d")(
            reg, {"a": "a", "b": "b", "d": math.inf}
        )


@pytest.mark.parametrize("n", [10**1000, math.inf, math.nan])
def test_bad_numeric_claims_are_reportable_in_strict_json(n):
    reg = FactRegistry()
    reg.record_table("t", [{"id": "a", "n": 1}], key="id", tool="query")
    with boundary(
        reg, facts=["t"], checks=[aggregate_reconciles(fact="t", field="n", column="n")]
    ) as b:
        report = b.report({"n": n})
    assert not report.ok
    json.dumps(report.to_dict(), allow_nan=False)


@pytest.mark.parametrize("tolerance", [-1, math.inf, math.nan, True])
def test_invalid_tolerance_is_a_configuration_error(tolerance):
    reg = FactRegistry()
    reg.record("f", 1, tool="query")
    reg.record_table("t", [{"id": "a", "n": 1}], key="id", tool="query")
    with pytest.raises(ValueError, match="tolerance"):
        row_integrity(fact="t", key_field="id", fields={"n": "n"}, tolerance=tolerance)(
            reg, {"id": "a", "n": 999}
        )


@pytest.mark.parametrize("k", [-1, 0, True, 1.0])
def test_imperative_and_builder_top_k_reject_invalid_config(k):
    reg = FactRegistry()
    reg.record_ranking("r", {"a": 1}, key="n", tool="query")
    with pytest.raises(ValueError, match="k must"):
        ranking_prefix(fact="r", field="top", k=k)
    with pytest.raises(ValueError, match="k must"):
        check_ranking_prefix(reg, {"top": ["a"]}, fact="r", field="top", k=k)


def test_invalid_boundary_mode_is_not_silently_treated_as_first():
    reg = FactRegistry()
    with pytest.raises(ValueError, match="mode"):
        boundary(reg, mode="typo")
    with boundary(reg, require_output=False) as b, pytest.raises(ValueError, match="mode"):
        b.submit({}, mode="")


@pytest.mark.parametrize(
    "op,values,weights",
    [
        ("sum", [2**53 + 1, 0.0], None),
        ("median", [2**53 + 1, 2**53 + 1], None),
        ("mean", [2**53 + 1], [1]),
    ],
)
def test_aggregate_intermediates_do_not_round_away_an_integer_unit(op, values, weights):
    reg = FactRegistry()
    rows = [{"id": i, "n": v, "w": weights[i] if weights else 1} for i, v in enumerate(values)]
    reg.record_table("t", rows, key="id", tool="query")
    check = aggregate_reconciles(
        fact="t", field="n", column="n", op=op, weight_column="w" if weights else None
    )
    check(reg, {"n": 2**53 + 1})
    with pytest.raises(UnsupportedClaim):
        check(reg, {"n": 2**53})


def test_weight_product_underflow_cannot_turn_a_nonzero_mean_into_zero():
    reg = FactRegistry()
    reg.record_table("t", [{"id": 1, "n": 1e-308, "w": 1e-308}], key="id", tool="query")
    check = aggregate_reconciles(fact="t", field="n", column="n", op="mean", weight_column="w")
    check(reg, {"n": 1e-308})
    with pytest.raises(UnsupportedClaim):
        check(reg, {"n": 0})


@pytest.mark.parametrize("kind,multiplier", [("abs", 1), ("pp", 100)])
def test_mixed_float_and_integer_comparison_preserves_the_integer(kind, multiplier):
    reg = FactRegistry()
    reg.record_ranking("r", {"a": 2**53 + 1, "b": 0.0}, key="n", tool="query")
    check = comparison(fact="r", a_field="a", b_field="b", delta_field="d", kind=kind)
    check(reg, {"a": "a", "b": "b", "d": (2**53 + 1) * multiplier})
    with pytest.raises(UnsupportedClaim):
        check(reg, {"a": "a", "b": "b", "d": (2**53) * multiplier})


def test_huge_comparison_mismatch_is_reported_without_diagnostic_overflow():
    reg = FactRegistry()
    reg.record_ranking("r", {"a": 10**1000, "b": 1}, key="n", tool="query")
    check = comparison(fact="r", a_field="a", b_field="b", delta_field="d")
    with pytest.raises(UnsupportedClaim):
        check(reg, {"a": "a", "b": "b", "d": 2})


def test_unrepresentable_nonzero_ratio_does_not_validate_zero():
    reg = FactRegistry()
    reg.record_ranking("r", {"a": 1e-308, "b": 1e308}, key="n", tool="query")
    check = comparison(fact="r", a_field="a", b_field="b", delta_field="d", kind="ratio")
    with pytest.raises(ValueError, match="floating-point range"):
        check(reg, {"a": "a", "b": "b", "d": 0})


def test_ordinary_float_roundoff_can_still_reconcile_with_an_integer_result():
    reg = FactRegistry()
    reg.record_table(
        "t", [{"id": i, "n": n} for i, n in enumerate([0.1, 0.2, 0.7])], key="id", tool="query"
    )
    aggregate_reconciles(fact="t", field="n", column="n")(reg, {"n": 1.0})
    reg.record_ranking("r", {"a": 0.3, "b": 0.1}, key="n", tool="query")
    comparison(fact="r", a_field="a", b_field="b", delta_field="d", kind="pp", tolerance=1e-9)(
        reg, {"a": "a", "b": "b", "d": 20.0}
    )


def test_unrepresentable_unweighted_diagnostic_does_not_veto_a_valid_weighted_mean():
    reg = FactRegistry()
    reg.record_table(
        "t",
        [{"id": 1, "n": 2**53, "w": 0}, {"id": 2, "n": 2**53 + 1, "w": 1}],
        key="id",
        tool="query",
    )
    aggregate_reconciles(fact="t", field="n", column="n", op="mean", weight_column="w")(
        reg, {"n": 2**53 + 1}
    )
    with pytest.raises(ValueError, match="precision"):
        aggregate_reconciles(fact="t", field="n", column="n", op="mean")(reg, {"n": 2**53})


def test_null_filter_does_not_select_rows_where_the_filter_column_is_missing():
    reg = FactRegistry()
    reg.record_table("t", [{"id": 1, "region": None}, {"id": 2}], key="id", tool="query")
    check = aggregate_reconciles(fact="t", field="n", op="count", where={"region": None})
    check(reg, {"n": 1})
    with pytest.raises(UnsupportedClaim):
        check(reg, {"n": 2})
