"""Bypasses found in the 2026-10 quality pass, each named for the failure it lets through."""

import time

import pytest

from groundplane import (
    Domain,
    FactBoundaryError,
    FactRegistry,
    Ranking,
    Table,
    UnsupportedClaim,
    boundary,
    field_matches_fact,
    superlative,
)
from groundplane.aggregates import check_aggregate_reconciles
from groundplane.entities import check_entities_recorded
from groundplane.rows import check_row_integrity

# --- hand-built shapes cannot skip the invariants record_* enforces -----------------


def test_hand_built_ranking_out_of_order_cannot_crown_the_runner_up():
    # Recorded through record(), an unsorted Ranking made "north" the winner even
    # though "harbour" scored higher: the ordering came from the caller, not code.
    with pytest.raises(ValueError, match="not ordered"):
        Ranking(key="ctr", items=(("north", 0.0412), ("harbour", 0.0455)))


def test_hand_built_lowest_wins_ranking_must_be_ascending():
    Ranking(key="ms", items=(("a", 10), ("b", 20)), higher_is_better=False)
    with pytest.raises(ValueError, match="not ordered"):
        Ranking(key="ms", items=(("a", 20), ("b", 10)), higher_is_better=False)


@pytest.mark.parametrize(
    "items,match",
    [
        ((), "empty"),
        ((("a", 2), ("a", 1)), "more than once"),
        ((("a", float("nan")),), "not finite"),
        ((("a", True),), "not a number"),
    ],
)
def test_hand_built_ranking_rejects_what_record_ranking_rejects(items, match):
    with pytest.raises(ValueError, match=match):
        Ranking(key="ctr", items=items)


def test_hand_built_table_with_duplicate_keys_cannot_answer_for_the_wrong_row():
    with pytest.raises(ValueError, match="more than once"):
        Table(key="id", rows=({"id": "a", "v": 1}, {"id": "a", "v": 99}), columns=("id", "v"))


@pytest.mark.parametrize(
    "rows,columns,match",
    [
        (({"id": "a"},), ("v",), "key column"),
        (({"v": 1},), ("id", "v"), "no key column"),
        ((), ("id",), "empty"),
    ],
)
def test_hand_built_table_rejects_what_record_table_rejects(rows, columns, match):
    with pytest.raises(ValueError, match=match):
        Table(key="id", rows=rows, columns=columns)


def test_hand_built_domain_holds_names_only():
    with pytest.raises(TypeError, match="not a name"):
        Domain(members=frozenset({"north", 101}))
    with pytest.raises(ValueError, match="empty"):
        Domain(members=frozenset())


def test_record_messages_still_name_the_fact():
    reg = FactRegistry()
    with pytest.raises(ValueError, match="'perf'"):
        reg.record_ranking("perf", [("a", 1), ("a", 2)], key="n", tool="q")
    with pytest.raises(ValueError, match="'rows'"):
        reg.record_table("rows", [{"id": 1}, {"id": 1}], key="id", tool="q")


# --- a boundary with nothing to check must not report a pass ------------------------


def test_boundary_without_checks_refuses_to_bless_output():
    reg = FactRegistry()
    reg.record_ranking("r", {"a": 2, "b": 1}, key="n", tool="q")
    with pytest.raises(FactBoundaryError, match="no checks"), boundary(reg, facts=["r"]) as b:
        b.submit({"winner": "b"})
    with pytest.raises(FactBoundaryError, match="no checks"), boundary(reg, facts=["r"]) as b:
        b.report({"winner": "b"})


def test_boundary_with_checks_still_submits():
    reg = FactRegistry()
    reg.record_ranking("r", {"a": 2, "b": 1}, key="n", tool="q")
    with boundary(reg, facts=["r"], checks=[superlative(fact="r")]) as b:
        assert b.submit({"winner": "a"}) == {"winner": "a"}


# --- Python's True == 1 alias does not leak through filters or composite names ------


def test_where_filter_on_a_flag_does_not_count_rows_flagged_with_integer_one():
    reg = FactRegistry()
    reg.record_table(
        "t",
        [{"id": "a", "flag": 1, "v": 10}, {"id": "b", "flag": True, "v": 5}],
        key="id",
        tool="q",
    )
    check_aggregate_reconciles(
        reg, {"total": 5}, fact="t", field="total", column="v", where={"flag": True}
    )
    with pytest.raises(UnsupportedClaim):
        check_aggregate_reconciles(
            reg, {"total": 15}, fact="t", field="total", column="v", where={"flag": True}
        )


def test_where_filter_still_matches_equal_numbers_across_int_and_float():
    reg = FactRegistry()
    reg.record_table("t", [{"id": "a", "tier": 1.0, "v": 3}], key="id", tool="q")
    check_aggregate_reconciles(reg, {"n": 3}, fact="t", field="n", column="v", where={"tier": 1})


def test_composite_key_with_a_bool_does_not_impersonate_an_int_member():
    reg = FactRegistry()
    reg.record_table("t", [{"id": (1, "x"), "v": 1}], key="id", tool="q")
    check_entities_recorded(reg, {"e": [(1, "x")]}, fields=["e"], fact="t")
    with pytest.raises(UnsupportedClaim):
        check_entities_recorded(reg, {"e": [(True, "x")]}, fields=["e"], fact="t")


def test_row_swap_diagnosis_does_not_blame_a_row_holding_one_for_true():
    reg = FactRegistry()
    reg.record_table(
        "t", [{"id": "a", "tags": [False]}, {"id": "b", "tags": [1]}], key="id", tool="q"
    )
    with pytest.raises(UnsupportedClaim, match="in no recorded row"):
        check_row_integrity(
            reg, {"id": "a", "tags": [True]}, fact="t", key_field="id", fields={"tags": "tags"}
        )


# --- large inputs --------------------------------------------------------------------


def test_field_matches_fact_on_a_large_mapping_is_not_quadratic():
    reg = FactRegistry()
    big = {f"k{i}": i for i in range(20_000)}
    reg.record("big", big, tool="q")
    started = time.perf_counter()
    field_matches_fact(field="x", fact="big")(reg, {"x": dict(big)})
    assert time.perf_counter() - started < 3.0


@pytest.mark.parametrize(
    "claimed,recorded,equal",
    [
        ({1: "x"}, {1: "x"}, True),
        ({True: "x"}, {1: "x"}, False),
        ({1.0: "x"}, {1: "x"}, False),
        ({"a": {"b": [1, 2]}}, {"a": {"b": [1, 2]}}, True),
        ({"a": {"b": [1, True]}}, {"a": {"b": [1, 1]}}, False),
        ({(1, "x"): 1}, {(True, "x"): 1}, False),
        ({1, 2}, {1, 2}, True),
        ({True, 2}, {1, 2}, False),
        ({"a": 1}, {"b": 1}, False),
    ],
)
def test_mapping_and_set_equality_keeps_key_types(claimed, recorded, equal):
    reg = FactRegistry()
    reg.record("f", recorded, tool="q")
    check = field_matches_fact(field="x", fact="f")
    if equal:
        check(reg, {"x": claimed})
    else:
        with pytest.raises(UnsupportedClaim):
            check(reg, {"x": claimed})
