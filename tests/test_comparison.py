"""Adversarial fixtures for comparison: deltas that are a difference, just not the one claimed.

Each case names the real-world failure it stands for. Uses the shared ``registry``
fixture from conftest: harbour 0.0455 vs north 0.0412 on CTR.
"""

import pytest

from groundplane import FactRegistry
from groundplane.comparison import check_comparison, comparison
from groundplane.errors import UnsupportedClaim

HARBOUR, NORTH = 0.0455, 0.0412
ABS = HARBOUR - NORTH  # 0.0043
PCT = ABS / NORTH * 100  # ~10.44 %
RATIO = HARBOUR / NORTH  # ~1.104
PP = ABS * 100  # 0.43 percentage points

OUT = {"a": "harbour", "b": "north"}


def _check(registry, delta, **kwargs):
    kwargs.setdefault("rel_tolerance", 1e-9)
    check_comparison(
        registry,
        {**OUT, "delta": delta},
        fact="campaign_ctr",
        a_field="a",
        b_field="b",
        delta_field="delta",
        **kwargs,
    )


# 1. The four conventions, each stated in its own terms, all reconcile.
@pytest.mark.parametrize(
    ("kind", "delta"),
    [("abs", ABS), ("pct", PCT), ("ratio", RATIO), ("pp", PP)],
)
def test_correct_delta_passes_for_each_kind(registry, kind, delta):
    _check(registry, delta, kind=kind)


def test_b_minus_a_direction_passes_with_swapped_sign(registry):
    _check(registry, -ABS, kind="abs", direction="b_minus_a")
    _check(registry, NORTH / HARBOUR, kind="ratio", direction="b_minus_a")


# 2. Right magnitude, wrong sign: the model subtracted in the other order.
def test_wrong_sign_diagnosed_as_reversed_direction(registry):
    with pytest.raises(UnsupportedClaim) as exc:
        _check(registry, -ABS, kind="abs")
    msg = str(exc.value)
    assert "claim matches b_minus_a; declared a_minus_b" in msg
    assert "warehouse.query" in msg
    assert exc.value.field == "delta"


def test_reversed_direction_diagnosed_for_pct(registry):
    reversed_pct = (NORTH - HARBOUR) / HARBOUR * 100
    with pytest.raises(UnsupportedClaim, match="claim matches b_minus_a; declared a_minus_b"):
        _check(registry, reversed_pct, kind="pct")


# 3. "Up 0.43%" when the gap is 0.43 percentage points (a 10.4% relative lift), and
#    the mirror image. Both are true numbers under the other convention.
def test_percentage_points_reported_as_percent_diagnosed(registry):
    with pytest.raises(UnsupportedClaim) as exc:
        _check(registry, PP, kind="pct")
    assert "equals the pp delta; field declared kind='pct'" in str(exc.value)


def test_percent_reported_as_percentage_points_diagnosed(registry):
    with pytest.raises(UnsupportedClaim) as exc:
        _check(registry, PCT, kind="pp")
    assert "equals the pct delta; field declared kind='pp'" in str(exc.value)


# 4. "1.10x" reported in a field declared as percent change.
def test_ratio_reported_as_pct_diagnosed(registry):
    with pytest.raises(UnsupportedClaim) as exc:
        _check(registry, RATIO, kind="pct")
    assert "equals the ratio delta; field declared kind='pct'" in str(exc.value)


# 5. A number that fits no convention gets the plain arithmetic, no false diagnosis.
def test_arbitrary_wrong_delta_reports_offset_without_diagnosis(registry):
    with pytest.raises(UnsupportedClaim) as exc:
        _check(registry, 12.0, kind="pct")
    msg = str(exc.value)
    assert "off by" in msg
    assert "equals the" not in msg
    assert "claim matches" not in msg


# 6. The model compares against a campaign that was never in the data.
def test_unknown_entity_raises_naming_known_entities(registry):
    with pytest.raises(UnsupportedClaim) as exc:
        check_comparison(
            registry,
            {"a": "harbour", "b": "atlas", "delta": 0.01},
            fact="campaign_ctr",
            a_field="a",
            b_field="b",
            delta_field="delta",
        )
    assert exc.value.field == "b"
    assert "'atlas' is not in the fact" in str(exc.value)
    assert "harbour" in str(exc.value)


# 7. "Harbour beat harbour": a self-comparison is not a claim about the data.
def test_same_entity_on_both_sides_raises(registry):
    with pytest.raises(UnsupportedClaim, match="compared with itself"):
        check_comparison(
            registry,
            {"a": "harbour", "b": "harbour", "delta": 0.0},
            fact="campaign_ctr",
            a_field="a",
            b_field="b",
            delta_field="delta",
        )


# 8. Table-backed: the compared figure is one column of a recorded row set.
def test_table_backed_comparison_via_column():
    reg = FactRegistry()
    reg.record_table(
        "spend",
        [
            {"campaign": "north", "spend": 88000, "ctr": 0.0412},
            {"campaign": "harbour", "spend": 61200, "ctr": 0.0455},
        ],
        key="campaign",
        tool="warehouse.query",
    )
    check_comparison(
        reg,
        {"a": "north", "b": "harbour", "delta": 26800},
        fact="spend",
        a_field="a",
        b_field="b",
        delta_field="delta",
        column="spend",
    )
    with pytest.raises(UnsupportedClaim, match="'spend'"):
        check_comparison(
            reg,
            {"a": "north", "b": "harbour", "delta": 26000},
            fact="spend",
            a_field="a",
            b_field="b",
            delta_field="delta",
            column="spend",
        )


# 9. Integer row ids are compared as integers; "101" is not row 101.
def test_int_keys_stay_int():
    reg = FactRegistry()
    reg.record_table(
        "stores",
        [{"id": 101, "sales": 250.0}, {"id": 102, "sales": 200.0}],
        key="id",
        tool="warehouse.query",
    )
    check_comparison(
        reg,
        {"a": 101, "b": 102, "delta": 1.25},
        fact="stores",
        a_field="a",
        b_field="b",
        delta_field="delta",
        column="sales",
        kind="ratio",
    )
    with pytest.raises(UnsupportedClaim, match="'101' is not in the fact"):
        check_comparison(
            reg,
            {"a": "101", "b": 102, "delta": 1.25},
            fact="stores",
            a_field="a",
            b_field="b",
            delta_field="delta",
            column="sales",
            kind="ratio",
        )


# 10. Percent change from zero is undefined; no claimed number can be right.
@pytest.mark.parametrize("kind", ["pct", "ratio"])
@pytest.mark.parametrize("delta", [0.0, 100.0, float("inf"), "n/a"])
def test_division_by_zero_base_raises_whatever_the_claim(kind, delta):
    reg = FactRegistry()
    reg.record_ranking("signups", {"new": 40, "old": 0}, key="signups", tool="warehouse.query")
    with pytest.raises(UnsupportedClaim, match="undefined whatever the claim"):
        check_comparison(
            reg,
            {"a": "new", "b": "old", "delta": delta},
            fact="signups",
            a_field="a",
            b_field="b",
            delta_field="delta",
            kind=kind,
        )
    # ...but the absolute gap is perfectly well defined.
    check_comparison(
        reg,
        {"a": "new", "b": "old", "delta": 40},
        fact="signups",
        a_field="a",
        b_field="b",
        delta_field="delta",
        kind="abs",
    )


# 11. Rounding: "about 10%" is only acceptable if the caller said so.
def test_rounded_pct_fails_at_zero_tolerance_and_passes_in_tolerance(registry):
    with pytest.raises(UnsupportedClaim):
        _check(registry, 10.4, kind="pct", rel_tolerance=None)
    _check(registry, 10.4, kind="pct", rel_tolerance=None, tolerance=0.05)


# 12. A prose delta ("about 10%") in a numeric field is a type confusion, not a number.
@pytest.mark.parametrize("delta", ["10%", None, True])
def test_non_numeric_claim_raises(registry, delta):
    with pytest.raises(UnsupportedClaim, match="claimed delta is not a number"):
        _check(registry, delta, kind="pct")


# 13. Wiring mistakes fail at construction, where a developer sees them.
def test_unknown_kind_raises_at_builder_time():
    with pytest.raises(ValueError, match="unknown kind"):
        comparison(fact="f", a_field="a", b_field="b", delta_field="d", kind="delta")  # type: ignore[arg-type]


def test_column_on_ranking_fact_raises_value_error(registry):
    with pytest.raises(ValueError, match="already a Ranking"):
        _check(registry, ABS, column="ctr")


def test_builder_runs_as_a_boundary_check(registry):
    check = comparison(
        fact="campaign_ctr",
        a_field="a",
        b_field="b",
        delta_field="delta",
        kind="pp",
        rel_tolerance=1e-9,
    )
    check(registry, {**OUT, "delta": PP})
    with pytest.raises(UnsupportedClaim):
        check(registry, {**OUT, "delta": ABS})
